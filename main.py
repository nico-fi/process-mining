from pm import Miner

miner = Miner('data/running-example.xes')
print('Converting event log into event stream...')
miner.xes_to_csv()
print('Processing event stream...')
variants = miner.read_stream()
miner.show_histogram()
