import os.path
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms

from LFLSeg_dataset import LFLSegDataset

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="01")
parser.add_argument("--epoch", type=int, default=100)

# Input of LFLSeg module is 224x224
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=128)

# Path to train/test dataset (txt files)
parser.add_argument("--train", type=str, default='data_path/train.txt')
parser.add_argument("--test", type=str, default='data_path/test.txt')

parser.add_argument("--modelname", type=str, default='resnet101_LFLSeg_v1')
parser.add_argument("--output", type=str, default='./trained_models/')


args = parser.parse_args()

gpu_list = ','.join(str(x) for x in args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

TRAIN = 'train'
TEST = 'test'

train_dataset = args.train
test_dataset = args.test

print("Train data: %s" % train_dataset)
print("Test data: %s" % test_dataset)

# Save trained models (classifiers)
save_folder = args.output

print('Save trained models to: ' + save_folder)

# Class: ['full_leaf': 0, 'partial_leaf': 1, 'non_leaf': 2]

def train_model(model, log_filename, optimizer, criterion, scheduler, dataloaders, num_epochs=args.epoch):
	since = time.time()

	best_acc = 0.0
	log_file = open(os.path.join(save_folder,'train_log_' + log_filename + '.txt'), 'w')

	for epoch in range(1, num_epochs+1):
		print('Epoch {}/{}'.format(epoch, num_epochs))
		print('-' * 20)

		log_file.write('Epoch {}/{}'.format(epoch, num_epochs) + '\n')
		log_file.write('-' * 20 + '\n')

		# Each epoch has a training and test phase
		for phase in [TRAIN, TEST]:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0.0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to('cuda')
				labels = labels.to('cuda')

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)

					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			log_file.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc) + '\n')

			
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc

				## Saving model
				if isinstance(model, nn.DataParallel):
					temp_model = model.module
				
				state_dict = temp_model.state_dict()
				for key, param in state_dict.items():
					state_dict[key] = param.cpu()
				torch.save(state_dict, os.path.join(save_folder, str(epoch) + '_best_model_' + log_filename + '.pth'))
				print("Saved best_model at epoch {}".format(epoch))
				log_file.write('Saved best_model at epoch {}\n'.format(epoch))

		# Saving model every 10 epoch
		if(epoch % 10 == 0):
			if isinstance(model, nn.DataParallel):
				temp_model = model.module
					
			state_dict = temp_model.state_dict()
			for key, param in state_dict.items():
				state_dict[key] = param.cpu()
			torch.save(state_dict, os.path.join(save_folder, 'trained_' + log_filename + '_%d.pth' % epoch))
			print()
			log_file.write('\n')

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))

def main():
	data_transforms = {
		TRAIN: transforms.Compose([
			transforms.RandomResizedCrop(size=args.input_size, scale=(0.8, 1.0), interpolation=3),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]),
		
		TEST: transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	}

	image_datasets = {
		TRAIN: LFLSegDataset(
			txt_path=train_dataset,
			transform=data_transforms[TRAIN]
		),
		TEST: LFLSegDataset(
			txt_path=test_dataset,
			transform=data_transforms[TEST]
		)
	}

	dataloaders = {
		x: torch.utils.data.DataLoader(
			image_datasets[x], batch_size=args.batch_size,
			shuffle=True, num_workers=32
		)
		for x in [TRAIN, TEST]
	}


	model_ft = models.resnet101(pretrained=True)
	
	model_name = args.modelname
	print(model_name)
	print(train_dataset)
	print("Number of epoch: %d"%args.epoch)

	# Replace final layer with 3 outputs (full leaf, partial leaf, non-leaf)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 3)

	for param in model_ft.parameters():
		param.requires_grad = True

	model_ft = nn.DataParallel(model_ft)
	model_ft = model_ft.to('cuda')

	criterion = nn.CrossEntropyLoss().to('cuda')

	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	train_model(model_ft, model_name, optimizer_ft, criterion, exp_lr_scheduler, dataloaders)


if __name__ == '__main__':
	main()
