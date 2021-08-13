from os import path
from pm import Miner

miner = Miner(path.join('eventlog', 'BPIC1.xes'))
miner.import_xes_log()
miner.generate_csv()
algorithm = miner.algo.IM
cut_point = 100
top_variants = 5
miner.process_csv_stream(algorithm, cut_point, top_variants)
miner.save_results(f'-{algorithm.name}.{cut_point}.{top_variants}')
miner.show_results()
miner.show_process_model()
miner.show_variant_histogram()
