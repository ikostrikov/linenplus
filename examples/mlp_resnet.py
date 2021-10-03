import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state

import linenplus as nnp

tf.config.experimental.set_visible_devices([], "GPU")
from typing import Any

import jax.numpy as jnp
import tqdm


def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_augemntation(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.uint8(train_ds['image'])
    test_ds['image'] = jnp.uint8(test_ds['image'])
    return train_ds, test_ds


@jax.jit
def update_model(rng, state, images, labels):
    key, rng = jax.random.split(rng)

    images = batched_augemntation(key, images)
    images = images.astype(jnp.float32) / 255.0
    images = jnp.reshape(images, [-1, 28 * 28])

    def loss_fn(params):
        logits = state.apply_fn({
            'params': params,
        }, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return loss, accuracy

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return rng, state, loss, accuracy


@jax.jit
def evaluate_model(state, images, labels):
    images = images.astype(jnp.float32) / 255.0
    images = jnp.reshape(images, [-1, 28 * 28])

    logits = state.apply_fn({
        'params': state.params,
    }, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def create_state(learning_rate, momentum, rng):
    model = nnp.mlp_resnet_v2.MLPResNetV2(num_blocks=10, num_classes=10)
    variables = model.init(rng, jnp.ones([1, 28 * 28]))
    params = variables['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params,
                                         tx=tx)


def compute_metrics(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(
        optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def train_epoch(state, train_ds, batch_size, rng, epoch):
    train_ds_size = len(train_ds['image'])
    steps_per_epochs = train_ds_size // batch_size
    permutations = jax.random.permutation(rng, train_ds_size)
    permutations = permutations[:steps_per_epochs * batch_size]
    permutations = permutations.reshape(steps_per_epochs, batch_size)

    batch_metrics = []
    for prm in tqdm.tqdm(permutations):
        batch_images = train_ds['image'][prm]
        batch_labels = train_ds['label'][prm]
        rng, state, loss, acc = update_model(rng, state, batch_images,
                                             batch_labels)
        metrics = {
            'loss': loss,
            'accuracy': acc,
        }
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)

    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' %
          (epoch + 1, epoch_metrics_np['loss'],
           epoch_metrics_np['accuracy'] * 100))

    return state


learning_rate = 0.1
momentum = 0.9
batch_size = 128
num_updates = 60000

train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate_fn = optax.piecewise_constant_schedule(learning_rate, {
    32000: 1e-1,
    48000: 1e-1
})
state = create_state(learning_rate_fn, momentum, init_rng)

num_epochs = round(0.5 + num_updates / (len(train_ds['image']) / batch_size))
for epoch in range(num_epochs):
    rng, input_rng = jax.random.split(rng)
    state = train_epoch(state, train_ds, batch_size, input_rng, epoch)
    test_loss, test_accuracy = evaluate_model(state, test_ds['image'],
                                              test_ds['label'])
    print(' test epoch: %d, loss: %.2f, accuracy: %.2f' %
          (epoch + 1, test_loss, test_accuracy * 100))
