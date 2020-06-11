import math
from collections import OrderedDict
from itertools import chain
import numpy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from code.base import ImageTranslatorBase
from code.models.feedforward import FeedForward
from code.models.masking import Masking
from code.models.unet import UNet
from code.optimisers.esadam import ESAdam
from code.utils.array.nd import extract_tiles
from code.utils.log.log import lprint, lsection


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class PTCNNImageTranslator(ImageTranslatorBase):
    """
        Pytorch-based CNN image translator
    """

    def __init__(
            self,
            max_epochs=2048,
            patience=None,
            patience_epsilon=0.0,
            learning_rate=0.01,
            batch_size=8,
            model_class=UNet,
            masking=True,
            masking_density=0.01,
            loss='l1',
            normaliser_type='percentile',
            balance_training_data=None,
            keep_ratio=1,
            max_voxels_for_training=4e6,
            monitor=None,
            use_cuda=True,
            device_index=0,
    ):
        """
        Constructs an image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(normaliser_type, monitor=monitor)

        use_cuda = use_cuda and (torch.cuda.device_count() > 0)
        self.device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
        lprint(f"Using device: {self.device}")

        self.max_epochs = max_epochs
        self.patience = max_epochs if patience is None else patience
        self.patience_epsilon = patience_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss = loss
        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data

        self.model_class = model_class

        self.l1_weight_regularisation = 1e-6
        self.l2_weight_regularisation = 1e-6
        self.training_noise = 0.1
        self.reload_best_model_period = max_epochs  # //2
        self.reduce_lr_patience = patience // 2
        self.reduce_lr_factor = 0.9
        self.masking = masking
        self.masking_density = masking_density
        self.optimiser_class = ESAdam
        self.max_tile_size = 1024  # TODO: adjust based on available memory

        self._stop_training_flag = False

    def _train(
            self,
            input_image,
            target_image,
            train_valid_ratio=0.1,
            callback_period=3,
            jinv=False,
    ):
        self._stop_training_flag = False

        if jinv is not None and not jinv:
            self.masking = False

        shape = input_image.shape
        num_batches = shape[0]
        num_input_channels = input_image.shape[1]
        num_output_channels = target_image.shape[1]
        num_spatiotemp_dim = input_image.ndim - 2

        # tile size:
        tile_size = min(self.max_tile_size, min(shape[2:]))

        # Decide on how many voxels to be used for validation:
        num_val_voxels = int(train_valid_ratio * input_image.size)
        lprint(
            f"Number of voxels used for validation: {num_val_voxels} (train_valid_ratio={train_valid_ratio})"
        )

        # Generate random coordinates for these voxels:
        val_voxels = tuple(numpy.random.randint(d, size=num_val_voxels) for d in shape)
        lprint(f"Validation voxel coordinates: {val_voxels}")

        # Training Tile size:
        lprint(f"Train Tile dimensions: {tile_size}")

        # Prepare Training Dataset:
        dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tilesize=tile_size,
            mode='grid',
            validation_voxels=val_voxels,
            batch_size=self.batch_size
        )
        lprint(f"Number tiles for training: {len(dataset)}")

        # Training Data Loader:
        # num_workers = max(3, os.cpu_count() // 2)
        num_workers = 0  # faster if data is already in memory...
        lprint(f"Number of workers for loading training/validation data: {num_workers}")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model
        self.model = self.model_class(num_input_channels, num_output_channels).to(self.device)

        number_of_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        lprint(
            f"Number of trainable parameters in {self.model_class} model: {number_of_parameters}"
        )

        if self.masking:
            self.masked_model = Masking(self.model, density=0.5).to(self.device)

        lprint(f"Optimiser class: {self.optimiser_class}")
        lprint(f"Learning rate : {self.learning_rate}")

        # Optimiser:
        optimizer = self.optimiser_class(
            chain(self.model.parameters()),
            lr=self.learning_rate,
            start_noise_level=self.training_noise,
            weight_decay=self.l2_weight_regularisation,
        )

        lprint(f"Optimiser: {optimizer}")

        # Denoise loss functon:
        loss_function = nn.L1Loss()
        if self.loss.lower() == 'l2':
            lprint(f"Training/Validation loss: L2")
            if self.masking:
                loss_function = (
                    lambda u, v, m: (u - v) ** 2 if m is None else ((u - v) * m) ** 2
                )
            else:
                loss_function = lambda u, v: (u - v) ** 2

        elif self.loss.lower() == 'l1':
            if self.masking:
                loss_function = (
                    lambda u, v, m: torch.abs(u - v)
                    if m is None
                    else torch.abs((u - v) * m)
                )
            else:
                loss_function = lambda u, v: torch.abs(u - v)
            lprint(f"Training/Validation loss: L1")

        # Start training:
        self._train_loop(data_loader, optimizer, loss_function)

    def _get_dataset(
            self,
            input_image: numpy.ndarray,
            target_image: numpy.ndarray,
            self_supervised: bool,
            tilesize: int,
            mode: str,
            validation_voxels,
            batch_size=32
    ):
        class _Dataset(Dataset):
            def __init__(self, input_image, target_image, tilesize):
                """
                """

                if batch_size > 1:
                    input_image = numpy.concatenate([input_image for _ in range(16)], axis=0)
                    target_image = numpy.concatenate([target_image for _ in range(16)], axis=0)

                num_channels_input = input_image.shape[1]
                num_channels_target = target_image.shape[1]

                def extract(image):
                    return extract_tiles(
                        image,
                        tile_size=tilesize,
                        extraction_step=tilesize,
                        flatten=True,
                    )

                bc_flat_input_image = input_image.reshape(-1, *input_image.shape[2:])
                bc_flat_input_tiles = numpy.concatenate(
                    [extract(x) for x in bc_flat_input_image]
                )
                self.input_tiles = bc_flat_input_tiles.reshape(
                    -1, num_channels_input, *bc_flat_input_tiles.shape[1:]
                )

                if self_supervised:
                    self.target_tiles = self.input_tiles
                else:
                    bc_flat_target_image = target_image.reshape(
                        -1, *target_image.shape[2:]
                    )
                    bc_flat_target_tiles = numpy.concatenate(
                        [extract(x) for x in bc_flat_target_image]
                    )
                    self.target_tiles = bc_flat_target_tiles.reshape(
                        -1, num_channels_target, *bc_flat_target_tiles.shape[1:]
                    )

                mask_image = numpy.zeros_like(input_image)
                mask_image[validation_voxels] = 1

                bc_flat_mask_image = mask_image.reshape(-1, *mask_image.shape[2:])
                bc_flat_mask_tiles = numpy.concatenate(
                    [extract(x) for x in bc_flat_mask_image]
                )
                self.mask_tiles = bc_flat_mask_tiles.reshape(
                    -1, num_channels_input, *bc_flat_mask_tiles.shape[1:]
                )

            def __len__(self):
                if batch_size > 1:
                    return 1
                else:
                    return len(self.input_tiles)

            def __getitem__(self, index):
                if batch_size > 1:
                    input = self.input_tiles[0, ...]
                    target = self.target_tiles[0, ...]
                    mask = self.mask_tiles[0, ...]
                else:
                    input = self.input_tiles[index, ...]
                    target = self.target_tiles[index, ...]
                    mask = self.mask_tiles[index, ...]

                return (input, target, mask)

        if mode == 'grid':
            return _Dataset(input_image, target_image, tilesize)
        else:
            return None

    def _train_loop(self, data_loader, optimizer, loss_function):

        # Scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.reduce_lr_factor,
            verbose=True,
            patience=self.reduce_lr_patience,
        )

        best_val_loss_value = math.inf
        best_model_state_dict = None
        patience_counter = 0

        with lsection(f"Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):

                    if hasattr(self, 'masked_model'):
                        self.masked_model.density = 0.005 * self.masking_density + 0.995 * self.masked_model.density
                        lprint(f"masking density: {self.masked_model.density}")

                    train_loss_value = 0
                    val_loss_value = 0
                    iteration = 0
                    for i, (input_images, target_images, val_mask_images) in enumerate(
                            data_loader
                    ):

                        lprint(f"index: {i}, shape:{input_images.shape}")

                        input_images_gpu = input_images.to(
                            self.device, non_blocking=True
                        )
                        target_images_gpu = target_images.to(
                            self.device, non_blocking=True
                        )
                        validation_mask_images_gpu = val_mask_images.to(
                            self.device, non_blocking=True
                        )

                        # Adding training noise to input:
                        if self.training_noise > 0:
                            with torch.no_grad():
                                alpha = self.training_noise / (
                                        1 + (10000 * epoch / self.max_epochs)
                                )
                                lprint(f"Training noise level: {alpha}")
                                training_noise = alpha * torch.randn_like(input_images)
                                input_images_gpu += training_noise.to(
                                    input_images_gpu.device
                                )

                        # Clear gradients w.r.t. parameters
                        optimizer.zero_grad()

                        # Forward pass:
                        self.model.train()
                        if self.masking:
                            translated_images_gpu = self.masked_model(input_images_gpu)
                        else:
                            translated_images_gpu = self.model(input_images_gpu)

                        # apply forward model:
                        forward_model_images_gpu = self._forward_model(
                            translated_images_gpu
                        )

                        # validation masking:
                        u = forward_model_images_gpu * (1 - validation_mask_images_gpu)
                        v = target_images_gpu * (1 - validation_mask_images_gpu)

                        # with napari.gui_qt():
                        #     viewer = napari.Viewer()
                        #     viewer.add_image(to_numpy(validation_mask_images_gpu), name='validation_mask_images_gpu')
                        #     viewer.add_image(to_numpy(forward_model_images_gpu), name='forward_model_images_gpu')
                        #     viewer.add_image(to_numpy(target_images_gpu), name='target_images_gpu')

                        # translation loss (per voxel):
                        if self.masking:
                            mask = self.masked_model.get_mask()
                            translation_loss = loss_function(u, v, mask)
                        else:
                            translation_loss = loss_function(u, v)

                        # loss value (for all voxels):
                        translation_loss_value = translation_loss.mean()

                        # Additional losses:
                        additional_loss_value = self._additional_losses(
                            translated_images_gpu, forward_model_images_gpu
                        )
                        if additional_loss_value is not None:
                            translation_loss_value += additional_loss_value

                        # backpropagation:
                        translation_loss_value.backward()

                        # Updating parameters
                        optimizer.step()

                        # post optimisation -- if needed:
                        self.model.post_optimisation()

                        # update training loss_deconvolution for whole image:
                        train_loss_value += translation_loss_value.item()
                        iteration += 1

                        # Validation:
                        with torch.no_grad():
                            # Forward pass:
                            self.model.eval()
                            if self.masking:
                                translated_images_gpu = self.masked_model(
                                    input_images_gpu
                                )
                            else:
                                translated_images_gpu = self.model(input_images_gpu)

                            # apply forward model:
                            forward_model_images_gpu = self._forward_model(
                                translated_images_gpu
                            )

                            # validation masking:
                            u = forward_model_images_gpu * validation_mask_images_gpu
                            v = target_images_gpu * validation_mask_images_gpu

                            # translation loss (per voxel):
                            if self.masking:
                                translation_loss = loss_function(u, v, None)
                            else:
                                translation_loss = loss_function(u, v)

                            # loss values:
                            translation_loss_value = (
                                translation_loss.mean().cpu().item()
                            )

                            # update validation loss_deconvolution for whole image:
                            val_loss_value += translation_loss_value
                            iteration += 1

                    train_loss_value /= iteration
                    lprint(f"Training loss value: {train_loss_value}")

                    val_loss_value /= iteration
                    lprint(f"Validation loss value: {val_loss_value}")

                    # Learning rate schedule:
                    scheduler.step(val_loss_value)

                    if val_loss_value < best_val_loss_value:
                        lprint(f"## New best val loss!")
                        if val_loss_value < best_val_loss_value - self.patience_epsilon:
                            lprint(f"## Good enough to reset patience!")
                            patience_counter = 0

                        # Update best val loss value:
                        best_val_loss_value = val_loss_value

                        # Save model:
                        best_model_state_dict = OrderedDict(
                            {k: v.to('cpu') for k, v in self.model.state_dict().items()}
                        )

                    else:
                        if (
                                epoch % max(1, self.reload_best_model_period) == 0
                                and best_model_state_dict
                        ):
                            lprint(f"Reloading best models to date!")
                            self.model.load_state_dict(best_model_state_dict)

                        if patience_counter > self.patience:
                            lprint(f"Early stopping!")
                            break

                        # No improvement:
                        lprint(
                            f"No improvement of validation losses, patience = {patience_counter}/{self.patience} "
                        )
                        patience_counter += 1

                    lprint(f"## Best val loss: {best_val_loss_value}")

                    if self._stop_training_flag:
                        lprint(f"Training interupted!")
                        break

        lprint(f"Reloading best models to date!")
        self.model.load_state_dict(best_model_state_dict)


    def _additional_losses(self, translated_image, forward_model_image):
        return None

    def _forward_model(self, input):
        return input

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """
            Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        with torch.no_grad():
            self.model.eval()
            input_image = torch.Tensor(input_image)
            input_image = input_image.to(self.device)
            inferred_image: torch.Tensor = self.model(input_image)
            inferred_image = inferred_image.detach().cpu().numpy()
            return inferred_image

    def visualise_weights(self):
        try:
            self.model.visualise_weights()
        except AttributeError:
            lprint(
                f"Method 'visualise_weights()' unavailable, cannot visualise weights. "
            )
