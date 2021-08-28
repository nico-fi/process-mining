from pm import Algo, Order, Miner

log_name = 'BPIC1'
Miner.generate_csv(log_name)
miner = Miner(log_name, Order.MIN, Algo.ILP, cut=100, top=5, filtering=False, frequency=False, update=False)
miner.process_stream()
miner.save_results()
miner.save_variant_histogram()
