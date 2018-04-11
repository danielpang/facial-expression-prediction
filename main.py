from data_loader.facial_data_loader import FacialDataLoader
from models.facial_model import FacialModel
from trainers.facial_trainer import FacialTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
	# capture the config path from the run arguments
	# then process the json configuration file
	try:
		args = get_args()
		config = process_config(args.config)
	except:
		print("missing or invalid arguments")
		exit(0)

	# create the experiments dirs
	create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

	print('Create the data generator.')
	data_loader = FacialDataLoader(config)

	print('Create the model.')
	model = FacialModel(config)

	print('Create the trainer')
	trainer = FacialTrainer(model.model, data_loader.get_train_data(), config)

	print('Start training the model.')
	trainer.train()


if __name__ == '__main__':
	main()
