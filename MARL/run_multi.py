from run_ma2c import train, train_hete, train_cl
import argparse
import random
import multiprocessing

def parse_args():
    """
    Description for this experiment:
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs.ini'
    parser = argparse.ArgumentParser(description=("Train multiple adversaries using DRL"))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")

    parser.add_argument('--runs', type=int, required=False,
                        default= 3)
    
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    
    parser.add_argument('--attack-ratio', type=float, required=False,
                        default=1.0, help="set the attack ratio")
    
    parser.add_argument('--train-param', type=str, required=False,
                        default="0", help="provide strategy parameters")
    args = parser.parse_args()
    return args

def worker(i):
    print("%i\n" % (i))



if __name__ == "__main__":
    args = parse_args()
    runs = args.runs
    progs =[]
    for i in range(runs):
        rand_seed = random.randint(0,1000)
        fun = train
        if args.option == "train_hete":
            fun = train_hete
        elif args.option == "train_cl":
            fun = train_cl
        prog = multiprocessing.Process(target=fun, args=(args, rand_seed))
        progs.append(prog)
        prog.start()

    for p in progs:
        p.join()

    print(" all run have executed")