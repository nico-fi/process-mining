import subprocess
from sys import stdout
from enum import Enum
from collections import Counter
from datetime import timedelta
from os import path, makedirs
from matplotlib import pyplot
from pandas import DataFrame
from tempfile import NamedTemporaryFile
from pm4py import read_bpmn, read_pnml
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.streaming.importer.csv import importer as csv_stream_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

CASE_ID_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
FINAL_ACTIVITY = '_END_'


class Algo(Enum):
    IND = 1
    SPL = 2
    ILP = 3


class Order(Enum):
    FRQ = 1
    MIN = 2
    MAX = 3


class Miner:

    @staticmethod
    def generate_csv(log_name, case_id=CASE_ID_KEY, activity=ACTIVITY_KEY, timestamp='time:timestamp'):
        """
        Converte il file XES in input in uno stream di eventi ordinati cronologicamente con formato CSV. Ogni traccia
        viene estesa con un evento conclusivo che ne definisca il termine
        :param log_name: nome del file XES (eventualmente compresso) contenente il log di eventi
        :param case_id: attributo identificativo dell'istanza di processo (con aggiunta del prefisso 'case:')
        :param activity: attributo identificativo dell'attività eseguita
        :param timestamp: attributo indicante l'istante di esecuzione di un evento
        """
        csv_path = path.join('eventlog', 'CSV', log_name + '.csv')
        if not path.isfile(csv_path):
            print('Generating CSV file from XES log...')
            xes_path = path.join('eventlog', 'XES', log_name)
            xes_path += '.xes.gz' if path.isfile(xes_path + '.xes.gz') else '.xes'
            log = xes_importer.apply(xes_path, variant=xes_importer.Variants.LINE_BY_LINE)
            for trace in log:
                trace.append({activity: FINAL_ACTIVITY, timestamp: trace[-1][timestamp] + timedelta(seconds=1)})
            dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
            dataframe = dataframe.filter(items=[activity, timestamp, case_id]).sort_values(timestamp, kind='mergesort')
            dataframe = dataframe.rename(columns={activity: ACTIVITY_KEY, case_id: CASE_ID_KEY})
            makedirs(path.dirname(csv_path), exist_ok=True)
            dataframe.to_csv(csv_path, index=False)

    def __init__(self, log_name, order, algorithm, use_filter, cut_point, top_variants):
        """
        Metodo costruttore
        :param log_name: nome del file CSV contenente lo stream di eventi
        :param order: criterio di ordinamento delle varianti
        :param algorithm: algoritmo da utilizzare nell'apprendimento di un nuovo modello
        :param use_filter: booleano per l'utilizzo di tecniche di filtering
        :param cut_point: numero di istanze da esaminare preliminarmente
        :param top_variants: numero di varianti da impiegare nella costruzione di un modello
        """
        self.log_name = log_name
        self.order = order
        self.algo = algorithm
        self.use_filter = use_filter
        self.cut_point = cut_point
        self.top_variants = top_variants
        self.processed_traces = 0
        self.variants = Counter()
        self.best_variants = None
        self.models = []
        self.drifts = []
        self.evaluations = []

    def process_stream(self):
        """
        Processa iterativamente uno stream di eventi in formato CSV, ignorando attività che si ripetano in modo
        consecutivo per un numero di occorrenze superiore a due e aggiornando il contatore delle varianti in
        corrispondenza di un evento finale. Dopo aver esaminato un dato numero di istanze preliminari, viene generato
        un primo modello di processo. La generazione di nuovi modelli avverrà in corrispondenza di un concept drift.
        Modello iniziale e modello corrente saranno impiegati nel calcolo di precision e fitness
        """
        print('\nProcessing event stream...')
        stream = csv_stream_importer.apply(path.join('eventlog', 'CSV', self.log_name + '.csv'))
        traces = {}
        for event in stream:
            case = event[CASE_ID_KEY]
            activity = event[ACTIVITY_KEY]
            if activity == FINAL_ACTIVITY:
                new_trace = tuple(traces.pop(case))
                self.variants[new_trace] += 1
                self.processed_traces += 1
                if self.processed_traces == self.cut_point:
                    self.select_best_variants()
                    self.drifts.append([self.processed_traces, *self.best_variants])
                    self.learn_model()
                elif self.processed_traces > self.cut_point:
                    stdout.write(f'\rCurrent model: {len(self.drifts)}\tCurrent trace: {self.processed_traces}')
                    self.evaluate_model(new_trace)
                    self.select_best_variants()
                    if self.best_variants != self.drifts[-1][1:]:
                        self.drifts.append([self.processed_traces, *self.best_variants])
                        self.learn_model()
            elif case not in traces:
                traces[case] = [activity]
            elif len(traces[case]) == 1 or traces[case][-1] != activity or traces[case][-2] != activity:
                traces[case].append(activity)

    def select_best_variants(self):
        """
        Seleziona le varianti più significative all'istante corrente, considerando un dato criterio d'ordine
        """
        if self.order == Order.FRQ:
            self.best_variants = list(item[0] for item in self.variants.most_common(self.top_variants))
        else:
            self.best_variants = list(item[0] for item in self.variants.most_common())
            self.best_variants.sort(key=lambda variant: len(variant), reverse=self.order == Order.MAX)
            self.best_variants = self.best_variants[:self.top_variants]

    def learn_model(self):
        """
        Genera un nuovo modello di processo impiegando le varianti più significative all'istante corrente
        """
        log = EventLog(Trace({ACTIVITY_KEY: activity} for activity in variant) for variant in self.best_variants)
        if self.algo == Algo.IND:
            variant = inductive_miner.Variants.IMf if self.use_filter else inductive_miner.Variants.IM
            self.models.append(inductive_miner.apply(log, variant=variant))
        else:
            with NamedTemporaryFile(suffix='.bpmn' if self.algo == Algo.SPL else '.pnml') as model_file:
                with NamedTemporaryFile(suffix='.xes') as log_file:
                    variant = xes_exporter.Variants.LINE_BY_LINE
                    xes_exporter.apply(log, log_file.name, variant, {variant.value.Parameters.SHOW_PROGRESS_BAR: False})
                    args = (path.join('scripts', 'run.sh'), self.algo.name, str(self.use_filter), log_file.name,
                            path.splitext(model_file.name)[0])
                    subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    if self.algo == Algo.SPL:
                        self.models.append(bpmn_converter.apply(read_bpmn(model_file.name)))
                    else:
                        self.models.append(read_pnml(model_file.name))

    def evaluate_model(self, trace):
        """
        Valuta modello iniziale e modello corrente sull'istanza di processo indicata
        :param trace: istanza di processo da impiegare nella valutazione
        """
        log = EventLog([Trace({ACTIVITY_KEY: activity} for activity in trace)])
        fitness_variant = fitness_evaluator.Variants.ALIGNMENT_BASED
        precision_variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
        precision_parameters = {precision_variant.value.Parameters.SHOW_PROGRESS_BAR: False}
        self.evaluations.append([])
        for model in (self.models[-1], self.models[0]):
            fitness = fitness_evaluator.apply(log, *model, variant=fitness_variant)['average_trace_fitness']
            precision = precision_evaluator.apply(log, *model, precision_parameters, precision_variant)
            self.evaluations[-1].extend([fitness, precision, 2 * fitness * precision / (fitness + precision)])

    def save_results(self):
        """
        Esporta concept drifts, valutazioni e modelli di processo
        """
        columns = ['current_trace'] + [f'trace_{i}' for i in range(1, self.top_variants + 1)]
        dataframe = DataFrame(self.drifts, columns=columns)
        dataframe.index.name = 'n°_training'
        folder = path.join('results', self.log_name, 'drift')
        makedirs(folder, exist_ok=True)
        file = f'{self.order.name}.{self.cut_point}.{self.top_variants}'
        dataframe.to_csv(path.join(folder, file + '.csv'))
        columns = ['fitness', 'precision', 'f-measure', 'static_fitness', 'static_precision', 'static_f-measure']
        dataframe = DataFrame(self.evaluations, columns=columns)
        dataframe.loc['MEAN'] = dataframe.mean()
        dataframe.index.name = 'n°_evaluation'
        folder = path.join('results', self.log_name, 'evaluation')
        makedirs(folder, exist_ok=True)
        filtered = 'UF' if self.use_filter else 'NF'
        file = f'{self.order.name}.{self.algo.name}.{filtered}.{self.cut_point}.{self.top_variants}'
        dataframe.to_csv(path.join(folder, file + '.csv'))
        folder = path.join('results', self.log_name, 'petri')
        makedirs(folder, exist_ok=True)
        for index, model in enumerate(self.models):
            pnml_exporter.apply(model[0], model[1], path.join(folder, file + f'-{index}.pnml'), model[2])

    def save_variant_histogram(self, y_log=False):
        """
        Esporta l'istogramma delle varianti di processo
        :param y_log: booleano per l'utilizzo di una scala logaritmica sull'asse delle ordinate
        """
        y_axis = self.variants.values() if self.order == Order.FRQ else [len(v) for v in self.variants.keys()]
        pyplot.bar(range(len(self.variants)), y_axis)
        pyplot.title(f'Traces processed: {self.processed_traces}     Variants found: {len(self.variants)}\n')
        pyplot.xlabel('Variants')
        pyplot.ylabel('Frequency' if self.order == Order.FRQ else 'Length')
        if y_log:
            pyplot.semilogy()
        file = ('frequency' if self.order == Order.FRQ else 'length') + '-histogram.png'
        pyplot.savefig(path.join('results', self.log_name, file))
