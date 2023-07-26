import argparse
import yaml

def parse_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/utd/student.yaml')
    parser.add_argument('--batch-size', type = int, default = 8, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')

    parser.add_argument('--epochs', type = int , default = 70, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')

    parser.add_argument('--base_lr', type = float, default = 0.001, metavar = 'LR', 
                        help = 'learning rate (default: 0.001)')

    parser.add_argument('--model' ,default= None, help = 'Name of Model to load')
    parser.add_argument('--model-args',action=DictAction,default=dict(), help='the arguments of model')
    parser.add_argument('--no-cuda', action = 'store_true', default = False, 
                        help = 'disables CUDA training')
    
    parser.add_argument('--seed', type =  int , default = 1 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many bathces to wait before logging training status')
    parser.add_argument('--val-bsize', type = int, default = 8, metavar = 'N', 
                        help = 'Validation batch sieze (default : 8)')
    parser.add_argument('--dataset', type = str, default= 'ncrc', metavar = 'D', help = 'Which dataset to use')

    parser.add_argument('--mocap', type = int, default = 600, help = 'Skeleton Frame number')

    parser.add_argument('--acc-frame', type = int , default = 150, help = 'Acceleton frame number')

    parser.add_argument('--num-joints', type = int, default = 29, help = 'Number of joints in skeleton data')

    parser.add_argument('--num-classes', type = int, default = 6, help = 'Number of classes in data')

    parser.add_argument('--fusion', type = str, default = 'simple', metavar = 'F', help = "Fusion method to choose (default : Simple)")

    return parser

if __name__ == "__main__":
    parser = parse_args()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()