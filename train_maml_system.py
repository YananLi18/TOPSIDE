from multiprocess.spawn import freeze_support

from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
if __name__ == '__main__':
    freeze_support()  # Fix a multiprocess bug
    args, device = get_args()
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(64, args.image_channels,
                                            args.image_height, args.image_width))
    # maybe_unzip_dataset(args=args)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
    # maml_system.evaluated_test_set_using_the_best_models(top_n_models=5)
