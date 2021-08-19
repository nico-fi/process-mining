import subprocess
from collections import Counter
from datetime import timedelta
from enum import Enum
from os import path, makedirs
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
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness
from pm4py.algo.evaluation.precision import algorithm as precision


class Algo(Enum):
    IND = 1
    ILP = 2
    SPL = 3


class Order(Enum):
    FRQ = 1
    MIN = 2
    MAX = 3


class Miner:

    def __init__(self, file_name, case_id_key='case:concept:name', activity_key='concept:name',
                 timestamp_key='time:timestamp'):
        """
        Metodo costruttore
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
        self.xes_log = None

    def import_xes_log(self):
        """
        Importa il log XES richiesto per la generazione del file CSV e la valutazione dei modelli di processo.
        L'ordinamento delle tracce ne riflette la data di conclusione
        """
        print('Importing XES log...')
        self.xes_log = xes_importer.apply(self.file_name, variant=xes_importer.Variants.LINE_BY_LINE)
        self.xes_log = sort_log(self.xes_log, key=lambda trace: trace[-1][self.timestamp_key])
        self.file_name = self.file_name.removesuffix('.gz').removesuffix('.xes')

    def generate_csv(self):
        """
        Genera uno stream di eventi ordinati cronologicamente (formato CSV) ricorrendo al log XES in memoria.
        Ogni traccia viene estesa con un evento conclusivo che ne definisca il termine
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

    def set_parameters(self, order, algorithm, cut_point, top_variants):
        """
        Definisce i parametri necessari all'elaborazione dello stream di eventi
        :param order: criterio di ordinamento delle varianti
        :param algorithm: algoritmo da utilizzare per apprendere un nuovo modello di processo
        :param cut_point: numero di istanze di processo da esaminare per apprendere e valutare un nuovo modello
        :param top_variants: numero di varianti da impiegare nella costruzione del modello
        """
        self.order = order
        self.algo = algorithm
        self.cut_point = cut_point
        self.top_variants = top_variants
        self.processed_traces = 0
        self.variants = Counter()
        self.net = None
        self.i_marking = None
        self.f_marking = None
        self.results = DataFrame(columns=['fitting_traces%', 'average_trace_fitness', 'precision', 'f-measure'])
        self.results.index.name = 'training'

    def process_csv_stream(self):
        """
        Processa iterativamente gli eventi archiviati in un file CSV, ignorando attività che si ripetano in modo
        consecutivo e aggiornando il contatore delle varianti in corrispondenza di un evento finale.
        Dopo aver esaminato un numero di tracce prestabilito, viene generato e valutato un nuovo modello di processo
        """
        stream = csv_stream_importer.apply(self.file_name + '.csv')
        traces = {}
        for event in stream:
            case_id = event[self.case_id_key]
            activity = event[self.activity_key]
            if activity == self.final_activity:
                entry = tuple(traces[case_id])
                self.variants[entry] = len(entry) if self.order == Order.MAX else -len(
                    entry) if self.order == Order.MIN else self.variants[entry] + 1
                del traces[case_id]
                self.processed_traces += 1
                if self.processed_traces % self.cut_point == 0:
                    self.learn_model()
                    self.evaluate_model()
            elif case_id not in traces:
                traces[case_id] = [activity]
            elif traces[case_id][-1] != activity:
                traces[case_id].append(activity)

    def learn_model(self):
        """
        Apprende un nuovo modello di processo utilizzando le varianti più significative all'istante corrente
        """
        print(f'Learning model {len(self.results) + 1}...')
        log = EventLog()
        best_variants = (item[0] for item in self.variants.most_common(self.top_variants))
        for trace in best_variants:
            log.append(Trace([{'concept:name': activity} for activity in trace]))
        if self.algo == Algo.IND:
            self.net, self.i_marking, self.f_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
        else:
            with NamedTemporaryFile(suffix=('.bpmn' if self.algo == Algo.SPL else '.pnml')) as model:
                with NamedTemporaryFile(suffix='.xes') as log_file:
                    variant = xes_exporter.Variants.LINE_BY_LINE
                    xes_exporter.apply(log, log_file.name, variant, {variant.value.Parameters.SHOW_PROGRESS_BAR: False})
                    args = ('scripts/run.sh', self.algo.name, log_file.name, path.splitext(model.name)[0])
                    subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                if self.algo == Algo.SPL:
                    self.net, self.i_marking, self.f_marking = bpmn_converter.apply(read_bpmn(model.name))
                else:
                    self.net, self.i_marking, self.f_marking = read_pnml(model.name)

    def evaluate_model(self, whole=False):
        """
        Valuta il modello di processo corrente su un numero prestabilito di istanze successive
        :param whole: booleano per richiedere la valutazione sull'intero log
        """
        print(f'Evaluating model {len(self.results) + 1}...')
        if whole or self.processed_traces < len(self.xes_log):
            metrics = {fitness: fitness.Variants.ALIGNMENT_BASED, precision: precision.Variants.ALIGN_ETCONFORMANCE}
            begin = 0 if whole else self.processed_traces
            end = len(self.xes_log) if whole else begin + self.cut_point
            for key, variant in metrics.items():
                parameters = {variant.value.Parameters.ACTIVITY_KEY: self.activity_key}
                metrics[key] = key.apply(self.xes_log[begin:end], self.net, self.i_marking, self.f_marking, parameters,
                                         variant)
            f_measure = 2 * metrics[fitness]['average_trace_fitness'] * metrics[precision] / (
                    metrics[fitness]['average_trace_fitness'] + metrics[precision])
            row = [metrics[fitness]['percentage_of_fitting_traces'], metrics[fitness]['average_trace_fitness'],
                   metrics[precision], f_measure]
        else:
            row = [nan, nan, nan, nan]
        self.results.loc['WHOLE' if whole else len(self.results) + 1] = row

    def save_results(self):
        """
        Salva la tabella di valutazione in formato CSV
        """
        if len(self.results) != 0:
            folder = path.join('results', path.split(self.file_name)[-1])
            makedirs(folder, exist_ok=True)
            name = f'{self.order.name}.{self.algo.name}.{self.cut_point}.{self.top_variants}'
            self.results.to_csv(path.join(folder, name + '.csv'))

    def show_results(self):
        """
        Mostra la tabella di valutazione
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
        Illustra l'istogramma delle varianti di processo
        :param y_log: booleano per l'utilizzo di una scala logaritmica sull'asse delle ordinate
        """
        pyplot.bar(range(len(self.variants)), [abs(value) for value in self.variants.values()])
        pyplot.title(f'Traces processed: {self.processed_traces}       Variants found: {len(self.variants)}\n')
        pyplot.xlabel('Variants')
        pyplot.ylabel('Frequency' if self.order == Order.FRQ else 'Length')
        if y_log:
            pyplot.semilogy()
        pyplot.show()
