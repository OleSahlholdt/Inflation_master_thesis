
import os
import shutil
import torch
from tqdm import tqdm
from exp.exp_main import Exp_Main
from default_args import Informer_default_args
import copy

Exp = Exp_Main
def run_experiment(args, seq_lengths, n_heads, encoder_layers, task_id):
    """
    Runs the experiment for a given task ID and prediction length.
    """
    args.task_id = task_id
    args.do_predict = True

    print(f"Pred length: {args.pred_len}")

    for model in ['Informer']:
        args.model = model
        for i in tqdm(range(0, 238)):
            if i == 0 or month == 12:
                best_loss, best_args, best_setting = perform_cross_validation(args, seq_lengths, n_heads, encoder_layers)
            # Create a new copy for each idx to avoid mutation issues
            current_args = copy.deepcopy(best_args)
            current_args.idx = i
            setting = train_best_model(current_args)
            if args.do_predict:
                perform_prediction(setting, current_args)


def perform_cross_validation(args, seq_lengths, n_heads, encoder_layers):
    """
    Performs cross-validation to find the best model configuration.
    """
    best_loss = float('inf')
    best_args, best_setting = None, None

    # Generate all combinations of kernel_size, seq_length, and label_length
    for heads, encoder_layer, seq_length, label_length in generate_combinations(seq_lengths, n_heads, encoder_layers):
        args.n_heads = heads
        args.e_layers = encoder_layer
        args.seq_len = seq_length
        args.label_len = label_length

        setting = create_experiment_setting(args)
        exp = Exp(args)
        print(f'>>>>>>>start cv : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        try:
            loss = exp.cross_validate(setting)
        except Exception as e:
            raise(e)
            print(f"===========================GOT ERROR: {e}")
            continue
        print(f"Validation Loss {loss}")
        torch.cuda.empty_cache()
        if loss < best_loss:
            print(f"New best model with loss: {loss}")
            best_loss = loss
            best_args = copy.deepcopy(args)
            best_setting = setting

    return best_loss, best_args, best_setting


def generate_combinations(seq_lengths, n_heads, encoder_layers):
    """
    Generates all valid combinations of kernel_size, seq_length, and label_length.
    """
    for heads in n_heads:
        for encoder_layer in encoder_layers:
            for idx, seq_length in enumerate(seq_lengths):
                for label_length in seq_lengths[:idx + 1]:
                    yield heads, encoder_layer, seq_length, label_length


def train_best_model(args):
    """
    Trains the best model configuration.
    """
    exp = Exp(args)
    setting = create_experiment_setting(args)
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')

    exp.train(setting)
    return setting


def perform_prediction(setting, args):
    """
    Performs prediction using the best model configuration.
    """
    print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp = Exp(args)
    exp.predict(setting, True)

    # Clean up checkpoints
    checkpoint_dir = r"Informer_checkpoints/"
    for ckpt in os.listdir(checkpoint_dir):
        shutil.rmtree(rf'{checkpoint_dir}/{ckpt}')


def create_experiment_setting(args):
    """
    Creates a unique experiment setting string based on the current arguments.
    """
    return '{}_{}_seqlen{}_labellen{}_heads{}_encoderlayers{}_dm{}'.format(
        args.idx,
        args.task_id,
        args.seq_len,
        args.label_len,
        args.n_heads,
        args.e_layers,
        args.d_model
    )

args = Informer_default_args()
args.batch_size = 8
args.train_epochs = 10

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    n_heads = [8, 16]
    encoder_layers = [2, 3, 4]
    seq_lengths = [12, 24, 36]
    horizon = f"h{args.pred_len}"
    run_experiment(args, seq_lengths, n_heads, encoder_layers, task_id=f"Informer_{horizon}")