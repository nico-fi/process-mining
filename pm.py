from collections import Counter
from datetime import timedelta
from matplotlib import pyplot
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.streaming.importer.csv import importer as csv_stream_importer


class Miner:

    def __init__(self, file_name, case_id_key='case:concept:name', activity_key='concept:name',
                 timestamp_key='time:timestamp', final_activity='END'):
        """
        :param file_name: nome del file XES o CSV contenente il log di eventi
        :param case_id_key: attributo identificativo dell'istanza di processo
        :param activity_key: attributo identificativo dell'attività svolta
        :param timestamp_key: attributo che denota l'istante di esecuzione di un evento
        :param final_activity: attività che sancisce il termine di un'istanza di processo
        """
        self.num_trainings = 0
        self.processed_traces = 0
        self.variants = Counter()
        self.file_name = file_name
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.final_activity = final_activity

    def xes_to_csv(self):
        """
        Converte un log di eventi (formato XES) in uno stream di eventi ordinati cronologicamente (formato CSV).
        Ciascuna traccia viene estesa con un evento conclusivo che ne sancisce il termine
        """
        log = xes_importer.apply(self.file_name, variant=xes_importer.Variants.LINE_BY_LINE)
        for trace in log:
            end_time = trace[-1][self.timestamp_key] + timedelta(seconds=1)
            trace.append({self.activity_key: self.final_activity, self.timestamp_key: end_time})
        dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        dataframe.index.name = 'index'
        dataframe.sort_values([self.timestamp_key, 'index'], inplace=True)
        self.file_name = self.file_name[:-3] + 'csv'
        dataframe.to_csv(self.file_name, index=False)

    def read_stream(self):
        """
        Processa iterativamente gli eventi archiviati in un file CSV,
        aggiornando il contatore delle varianti in corrispondenza di ogni evento conclusivo.
        La ripetizione di una stessa attività in eventi consecutivi viene ignorata
        """
        traces = {}
        stream = csv_stream_importer.apply(self.file_name)
        for event in stream:
            case_id = event[self.case_id_key]
            activity = event[self.activity_key]
            if activity == self.final_activity:
                self.variants[tuple(traces[case_id])] += 1
                self.processed_traces += 1
                del traces[case_id]
            elif case_id not in traces:
                traces[case_id] = [activity]
            elif traces[case_id][-1] != activity:
                traces[case_id].append(activity)

    def show_histogram(self, y_log=False):
        """
        Illustra l'istogramma delle varianti correntemente analizzate
        :param y_log: richiede l'adozione di una scala logaritmica sull'asse delle frequenze
        """
        pyplot.bar(range(len(self.variants)), self.variants.values())
        pyplot.title(f'Traces processed: {self.processed_traces}       Variants found: {len(self.variants)}\n')
        pyplot.xlabel('Variants')
        pyplot.ylabel('Frequency')
        if y_log:
            pyplot.semilogy()
        pyplot.show()
