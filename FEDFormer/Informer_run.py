
import os
import shutil
import torch
from tqdm import tqdm
from exp.exp_main import Exp_Main
from default_args import Informer_default_args
import copy
import optuna
import joblib

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
        for i in tqdm(range(0, 118)):
            args.idx = i
            month = (i % 12) + 1
            if i == 0 or month == 12:
                best_loss, best_args, best_setting, study = perform_cross_validation(args, seq_lengths, n_heads, encoder_layers)
            # Create a new copy for each idx to avoid mutation issues
            current_args = copy.deepcopy(best_args)
            current_args.idx = i
            setting = train_best_model(current_args)
            if args.do_predict:
                perform_prediction(setting, current_args)
            folder_path = './results/' + setting + '/'
            # save after tuning
            joblib.dump(study, folder_path + "optuna_study.pkl")


def objective(trial, args, seq_lengths, n_heads, encoder_layers):
    # === Architecture choices ===
    args.n_heads = trial.suggest_categorical("n_heads", n_heads)
    args.e_layers = trial.suggest_categorical("e_layers", encoder_layers)
    args.seq_len = trial.suggest_categorical("seq_len", seq_lengths)
    args.label_len = trial.suggest_categorical(
        "label_len", [l for l in seq_lengths if l <= args.seq_len]
    )

    # d_model must be divisible by n_heads in most implementations
    possible_d_models = [128, 256, 512]
    args.d_model = trial.suggest_categorical(
        "d_model", [dm for dm in possible_d_models if dm % args.n_heads == 0]
    )

    args.d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.3)

    # === Training hyperparameters ===
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # (optional) Decoder layers if your Exp_Main supports it
    # args.d_layers = trial.suggest_categorical("d_layers", [1, 2, 3])

    # === Run experiment ===
    setting = create_experiment_setting(args)
    exp = Exp(args)
    print(f'>>>>>>> start optuna trial: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>')

    try:
        loss = exp.cross_validate(setting)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")  # penalize failed trial

    torch.cuda.empty_cache()
    return loss


def perform_cross_validation(args, seq_lengths, n_heads, encoder_layers, n_trials=50):
    """
    Uses Optuna to find the best model configuration.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, args, seq_lengths, n_heads, encoder_layers),
        n_trials=n_trials,
    )

    best_loss = study.best_value
    best_params = study.best_params

    # Apply best params to args
    best_args = copy.deepcopy(args)
    for k, v in best_params.items():
        setattr(best_args, k, v)

    best_setting = create_experiment_setting(best_args)
    print(f"Best params: {best_params}, Best loss: {best_loss}")

    return best_loss, best_args, best_setting, study


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
    return (
        f"{args.idx}_{args.task_id}_"
        f"seqlen{args.seq_len}_labellen{args.label_len}_"
        f"heads{args.n_heads}_encoderlayers{args.e_layers}_"
        f"dm{args.d_model}_dff{args.d_ff}_"
        f"drop{args.dropout:.2f}_"
        f"lr{args.learning_rate:.0e}_"
        f"bs{args.batch_size}"
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