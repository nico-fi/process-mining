from collections import Counter
from datetime import timedelta
from os import path
from math import nan
from matplotlib import pyplot
from pandas import DataFrame
from pm4py import sort_log
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.streaming.importer.csv import importer as csv_stream_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py import view_petri_net


class Miner:

    def __init__(self, file_name, case_id_key='case:concept:name', activity_key='concept:name',
                 timestamp_key='time:timestamp'):
        """
        Costruttore
        :param file_name: nome del file XES (eventualmente compresso) contenente il log di eventi
        :param case_id_key: attributo identificativo dell'istanza di processo (con aggiunta del prefisso 'case:')
        :param activity_key: attributo identificativo dell'attività eseguita
        :param timestamp_key: attributo che denota l'istante di esecuzione di un evento
        """
        self.file_name = file_name
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.final_activity = 'END'
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
        L'ordinamento delle tracce ne riflette la data di conclusione
        """
        print('Importing XES log...')
        self.xes_log = xes_importer.apply(self.file_name, variant=xes_importer.Variants.LINE_BY_LINE)
        self.xes_log = sort_log(self.xes_log, key=lambda trace: trace[-1][self.timestamp_key])
        self.file_name = self.file_name[:-7] if self.file_name.endswith('gz') else self.file_name[:-4]

    def generate_csv(self):
        """
        Genera uno stream di eventi ordinati cronologicamente (formato CSV) partendo dal log in formato XES.
        Ogni istanza di processo viene estesa con un evento conclusivo che ne sancisca il termine
        """
        print('Generating CSV file from XES log...')
        dataframe = log_converter.apply(self.xes_log, variant=log_converter.Variants.TO_DATA_FRAME)
        final_events = []
        for trace in self.xes_log:
            end_time = trace[-1][self.timestamp_key] + timedelta(seconds=1)
            case_id = trace.attributes[self.case_id_key[5:]]
            event = {self.activity_key: self.final_activity, self.timestamp_key: end_time, self.case_id_key: case_id}
            final_events.append(event)
        dataframe = dataframe.append(final_events, ignore_index=True)
        dataframe.index.name = 'index'
        dataframe = dataframe.sort_values([self.timestamp_key, dataframe.index.name])
        dataframe.to_csv(self.file_name + '.csv', index=False)

    def read_stream_from_csv(self, cut_point, top_variants):
        """
        Processa iterativamente gli eventi archiviati in un file CSV, ignorando eventi consecutivi relativi a una
        medesima attività e aggiornando il contatore delle varianti in corrispondenza di ciascun evento conclusivo.
        Dopo aver esaminato un dato numero di istanze, le varianti di processo più frequenti vengono impiegate
        per apprendere un nuovo modello
        :param cut_point: numero di istanze da esaminare prima di apprendere e valutare un nuovo modello
        :param top_variants: numero di varianti più frequenti da impiegare nella costruzione del modello
        """
        traces = {}
        stream = csv_stream_importer.apply(self.file_name + '.csv')
        for event in stream:
            case_id = event[self.case_id_key]
            activity = event[self.activity_key]
            if activity == self.final_activity:
                self.variants[tuple(traces[case_id])] += 1
                del traces[case_id]
                self.processed_traces += 1
                if self.processed_traces % cut_point == 0:
                    self.learn_model(top_variants)
                    self.evaluate()
            elif case_id not in traces:
                traces[case_id] = [activity]
            elif traces[case_id][-1] != activity:
                traces[case_id].append(activity)

    def learn_model(self, top_variants):
        """
        Apprende un nuovo modello di processo utilizzando le varianti più frequenti all'istante corrente
        :param top_variants: numero di varianti più frequenti da impiegare nella costruzione del modello
        """
        print(f'Learning model {len(self.results) + 1}...')
        log = EventLog()
        frequent_variants = (item[0] for item in self.variants.most_common(top_variants))
        for trace in frequent_variants:
            log.append(Trace([{self.activity_key: activity} for activity in trace]))
        variant = inductive_miner.Variants.IM
        parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
        self.net, self.i_marking, self.f_marking = inductive_miner.apply(log, parameters, variant)

    def evaluate(self):
        """
        Valuta il modello corrente utilizzando le istanze di processo non ancora esaminate.
        Vengono calcolati: percentuale di fitting traces, average trace fitness, precision, f-measure
        """
        print(f'Evaluating model {len(self.results) + 1}...')
        if self.processed_traces == len(self.xes_log):
            new_row = [nan, nan, nan, nan]
        else:
            variant = fitness_evaluator.Variants.ALIGNMENT_BASED
            parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
            fitness = fitness_evaluator.apply(self.xes_log[self.processed_traces:], self.net, self.i_marking,
                                              self.f_marking, parameters, variant)
            variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
            parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
            precision = precision_evaluator.apply(self.xes_log[self.processed_traces:], self.net, self.i_marking,
                                                  self.f_marking, parameters, variant)
            f_measure = 2 * precision * fitness['average_trace_fitness'] / (
                    precision + fitness['average_trace_fitness'])
            new_row = [fitness['percentage_of_fitting_traces'], fitness['average_trace_fitness'], precision, f_measure]
        self.results.loc[len(self.results) + 1] = new_row

    def print_results(self):
        """
        Stampa la tabella dei risultati, fornendo la valutazione di ciascun modello appreso
        """
        if len(self.results) == 0:
            print('No evaluation performed!')
        else:
            print(self.results)

    def save_results(self):
        """
        Salva la tabella dei risultati in formato CSV, fornendo la valutazione di ciascun modello appreso
        """
        self.results.to_csv(path.join('results', path.split(self.file_name)[-1] + '.csv'))

    def show_process_model(self):
        """
        Illustra il modello di processo corrente come Rete di Petri
        """
        if self.net is None:
            print("No model learned!")
        else:
            view_petri_net(self.net, self.i_marking, self.f_marking, format='png')

    def show_variant_histogram(self, y_log=False):
        """
        Illustra l'istogramma di varianti di processo e relative frequenze
        :param y_log: booleano che specifica l'utilizzo di una scala logaritmica per l'asse delle frequenze
        """
        pyplot.bar(range(len(self.variants)), self.variants.values())
        pyplot.title(f'Traces processed: {self.processed_traces}       Variants found: {len(self.variants)}\n')
        pyplot.xlabel('Variants')
        pyplot.ylabel('Frequency')
        if y_log:
            pyplot.semilogy()
        pyplot.show()
