import subprocess
from collections import Counter
from datetime import timedelta
from enum import Enum
from os import path
from math import nan
from matplotlib import pyplot
from pandas import DataFrame
from tempfile import NamedTemporaryFile
from pm4py import sort_log, read_bpmn, read_pnml, view_petri_net
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.streaming.importer.csv import importer as csv_stream_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator


class Miner:

    def __init__(self, file_name, case_id_key='case:concept:name', activity_key='concept:name',
                 timestamp_key='time:timestamp'):
        """
        Costruttore
        :param file_name: nome del file XES (eventualmente compresso) contenente il log di eventi
        :param case_id_key: attributo identificativo dell'istanza di processo (con aggiunta del prefisso 'case:')
        :param activity_key: attributo identificativo dell'attività eseguita
        :param timestamp_key: attributo indicante l'istante di esecuzione di un evento
        """
        self.file_name = file_name
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.final_activity = 'END'
        self.algo = Enum('Algorithm', 'IM SPLIT ILP')
        self.processed_traces = 0
        self.variants = Counter()
        self.xes_log = None
        self.net = None
        self.i_marking = None
        self.f_marking = None
        self.results = DataFrame(columns=['fitting_traces%', 'average_trace_fitness', 'precision', 'f-measure'])
        self.results.index.name = 'n°_training'

    def import_xes_log(self):
        """
        Importa il log XES richiesto per la generazione di un file CSV e la valutazione dei modelli di processo.
        L'ordinamento delle tracce ne riflette la data conclusiva
        """
        print('Importing XES log...')
        self.xes_log = xes_importer.apply(self.file_name, variant=xes_importer.Variants.LINE_BY_LINE)
        self.xes_log = sort_log(self.xes_log, key=lambda trace: trace[-1][self.timestamp_key])
        self.file_name = self.file_name.removesuffix('.gz').removesuffix('.xes')

    def generate_csv(self):
        """
        Genera uno stream di eventi ordinati cronologicamente (formato CSV), partendo dal log XES in memoria.
        Ogni istanza di processo viene estesa con un evento conclusivo che ne definisca il termine
        """
        print('Generating CSV file from XES log...')
        dataframe = log_converter.apply(self.xes_log, variant=log_converter.Variants.TO_DATA_FRAME)
        final_events = []
        for trace in self.xes_log:
            end_time = trace[-1][self.timestamp_key] + timedelta(seconds=1)
            case_id = trace.attributes[self.case_id_key.removeprefix('case:')]
            event = {self.activity_key: self.final_activity, self.timestamp_key: end_time, self.case_id_key: case_id}
            final_events.append(event)
        dataframe = dataframe.append(final_events, ignore_index=True)
        dataframe.index.name = 'index'
        dataframe = dataframe.sort_values([self.timestamp_key, dataframe.index.name])
        dataframe.to_csv(self.file_name + '.csv', index=False)

    def process_csv_stream(self, algorithm, cut_point, top_variants):
        """
        Processa iterativamente gli eventi archiviati in un file CSV, ignorando attività che si ripetano in modo
        consecutivo e aggiornando il contatore delle varianti in corrispondenza di ciascun evento finale.
        Dopo aver esaminato un dato numero di istanze, le varianti di processo più frequenti vengono impiegate
        per apprendere un nuovo modello
        :param algorithm: algoritmo da utilizzare per apprendere il nuovo modello (IM, SPLIT, ILP)
        :param cut_point: numero di istanze da esaminare prima di apprendere e valutare il nuovo modello
        :param top_variants: numero di varianti più frequenti da impiegare nella costruzione del modello
        """
        stream = csv_stream_importer.apply(self.file_name + '.csv')
        traces = {}
        for event in stream:
            case_id = event[self.case_id_key]
            activity = event[self.activity_key]
            if activity == self.final_activity:
                self.variants[tuple(traces[case_id])] += 1
                del traces[case_id]
                self.processed_traces += 1
                if self.processed_traces % cut_point == 0:
                    self.learn_model(algorithm, top_variants)
                    self.evaluate(cut_point)
            elif case_id not in traces:
                traces[case_id] = [activity]
            elif traces[case_id][-1] != activity:
                traces[case_id].append(activity)

    def learn_model(self, algorithm, top_variants):
        """
        Apprende un nuovo modello di processo utilizzando le varianti più frequenti all'istante corrente
        :param algorithm: algoritmo utilizzato per apprendere il nuovo modello
        :param top_variants: numero di varianti più frequenti da impiegare nella costruzione del modello
        """
        print(f'Learning model {len(self.results) + 1}...')
        log = EventLog()
        frequent_variants = (item[0] for item in self.variants.most_common(top_variants))
        for trace in frequent_variants:
            log.append(Trace([{self.activity_key: activity} for activity in trace]))
        if algorithm == self.algo.IM:
            self.net, self.i_marking, self.f_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
        else:
            with NamedTemporaryFile(suffix=('.bpmn' if algorithm == self.algo.SPLIT else '.pnml')) as model:
                with NamedTemporaryFile(suffix='.xes') as log_file:
                    variant = xes_exporter.Variants.LINE_BY_LINE
                    xes_exporter.apply(log, log_file.name, variant, {variant.value.Parameters.SHOW_PROGRESS_BAR: False})
                    args = ('scripts/run.sh', algorithm.name, log_file.name, path.splitext(model.name)[0])
                    subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                if algorithm == self.algo.SPLIT:
                    self.net, self.i_marking, self.f_marking = bpmn_converter.apply(read_bpmn(model.name))
                else:
                    self.net, self.i_marking, self.f_marking = read_pnml(model.name)

    def evaluate(self, cut_point):
        """
        Valuta il modello di processo corrente utilizzando un numero prefissato di istanze successive.
        Vengono forniti: percentuale di fitting traces, average trace fitness, precision, f-measure
        :param cut_point: numero di istanze di processo successive da impiegare nella valutazione
        """
        print(f'Evaluating model {len(self.results) + 1}...')
        if self.processed_traces == len(self.xes_log):
            new_row = [nan, nan, nan, nan]
        else:
            variant = fitness_evaluator.Variants.ALIGNMENT_BASED
            parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
            fitness = fitness_evaluator.apply(self.xes_log[self.processed_traces:self.processed_traces + cut_point],
                                              self.net, self.i_marking, self.f_marking, parameters, variant)
            variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
            parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
            precision = precision_evaluator.apply(self.xes_log[self.processed_traces:self.processed_traces + cut_point],
                                                  self.net, self.i_marking, self.f_marking, parameters, variant)
            f_measure = 2 * precision * fitness['average_trace_fitness'] / (
                    precision + fitness['average_trace_fitness'])
            new_row = [fitness['percentage_of_fitting_traces'], fitness['average_trace_fitness'], precision, f_measure]
        self.results.loc[len(self.results) + 1] = new_row

    def save_results(self, specs=''):
        """
        Salva la tabella dei risultati in formato CSV, fornendo la valutazione di ciascun modello appreso
        :param specs: sequenza di caratteri da utilizzare nel nome del file
        """
        if len(self.results) != 0:
            self.results.to_csv(path.join('results', path.split(self.file_name)[-1] + specs + '.csv'))

    def show_results(self):
        """
        Mostra la tabella dei risultati, fornendo la valutazione di ciascun modello appreso
        """
        if len(self.results) != 0:
            print(self.results)
        else:
            print('No evaluation performed!')

    def show_process_model(self):
        """
        Illustra il modello di processo corrente come Rete di Petri
        """
        if self.net is not None:
            view_petri_net(self.net, self.i_marking, self.f_marking, format='png')
        else:
            print("No model learned!")

    def show_variant_histogram(self, y_log=False):
        """
        Illustra l'istogramma delle varianti di processo con relative frequenze
        :param y_log: booleano che specifica l'utilizzo di una scala logaritmica per l'asse delle frequenze
        """
        pyplot.bar(range(len(self.variants)), self.variants.values())
        pyplot.title(f'Traces processed: {self.processed_traces}       Variants found: {len(self.variants)}\n')
        pyplot.xlabel('Variants')
        pyplot.ylabel('Frequency')
        if y_log:
            pyplot.semilogy()
        pyplot.show()
