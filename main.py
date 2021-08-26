from pm import Algo, Order, Miner

log_name = 'BPIC1'
Miner.generate_csv(log_name)
miner = Miner(log_name, Order.FRQ, Algo.IND, use_filter=False, cut_point=100, top_variants=5)
miner.process_stream()
miner.save_results()
miner.save_variant_histogram()
