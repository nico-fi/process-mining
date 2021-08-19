from os import path
from pm import Order, Algo, Miner

miner = Miner(path.join('eventlog', 'BPIC1.xes'))
miner.import_xes_log()
miner.generate_csv()

miner.set_parameters(Order.FRQ, Algo.IND, cut_point=100, top_variants=5)
miner.process_csv_stream()
miner.learn_model()
miner.evaluate_model(whole=True)
miner.save_results()

miner.show_results()
miner.show_process_model()
miner.show_variant_histogram()
