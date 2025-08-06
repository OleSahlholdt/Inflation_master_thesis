
import os
import shutil
import torch
from tqdm import tqdm
from exp.exp_main import Exp_Main
from default_args import Autoformer_default_args

Exp = Exp_Main
def run_experiment(args, seq_lengths, kernel_sizes, task_id):
    """
    Runs the experiment for a given task ID and prediction length.
    """
    args.task_id = task_id
    args.do_predict = True

    print(f"Pred length: {args.pred_len}")

    for model in ['Autoformer']:
        args.model = model
        for i in tqdm(range(0, 238)):
            args.idx = i
            month = (i % 12) + 1
            if i == 0 or month == 12:
                best_loss, best_args, best_setting = perform_cross_validation(args, seq_lengths, kernel_sizes)
            best_args.idx = i
            train_best_model(best_args)

            if args.do_predict:
                perform_prediction(best_setting)


def perform_cross_validation(args, seq_lengths, kernel_sizes):
    """
    Performs cross-validation to find the best model configuration.
    """
    best_loss = float('inf')
    best_args, best_setting = None, None

    # Generate all combinations of kernel_size, seq_length, and label_length
    for kernel_size, seq_length, label_length in generate_combinations(seq_lengths, kernel_sizes):
        args.moving_avg = kernel_size
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
            best_args = args
            best_setting = setting

    return best_loss, best_args, best_setting


def generate_combinations(seq_lengths, kernel_sizes):
    """
    Generates all valid combinations of kernel_size, seq_length, and label_length.
    """
    for kernel_size in kernel_sizes:
        for idx, seq_length in enumerate(seq_lengths):
            for label_length in seq_lengths[:idx + 1]:
                yield kernel_size, seq_length, label_length


def train_best_model(args):
    """
    Trains the best model configuration.
    """
    exp = Exp(args)
    setting = create_experiment_setting(args)
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')

    exp.train(setting)


def perform_prediction(setting):
    """
    Performs prediction using the best model configuration.
    """
    print(f'>>>>>>>start prediction_training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp = Exp(args)
    exp.predict(setting, True)

    # Clean up checkpoints
    checkpoint_dir = r"Autoformer_checkpoints/"
    for ckpt in os.listdir(checkpoint_dir):
        shutil.rmtree(rf'{checkpoint_dir}/{ckpt}')


def create_experiment_setting(args):
    """
    Creates a unique experiment setting string based on the current arguments.
    """
    return '{}_{}_seqlen{}_labellen{}_heads{}_encoderlayers{}_dm{}_ma{}'.format(
        args.idx,
        args.task_id,
        args.seq_len,
        args.label_len,
        args.n_heads,
        args.e_layers,
        args.d_model,
        args.moving_avg,
    )

args = Autoformer_default_args()
args.batch_size = 8

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    seq_lengths = [6, 12, 24]
    kernel_sizes = [5, 9, 13]
    run_experiment(args, seq_lengths, kernel_sizes, task_id="Autoformer_h1")