from os import path
from pm import Miner

miner = Miner(path.join('data', 'Road_Traffic_Fine_Management_Process.xes.gz'))
miner.import_xes_log()
miner.generate_csv()
miner.read_stream_from_csv(cut_point=15000, top_variants=5)
miner.print_results()
miner.save_results()
miner.show_process_model()
miner.show_variant_histogram()
