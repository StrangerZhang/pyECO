import numpy as np
from .config import config

import ipdb as pdb

def find_gram_vector(samplesf, new_sample, num_training_samples):
    gram_vector = np.inf * np.ones((config.num_samples))
    num_feature_blocks = len(new_sample)
    if num_training_samples == config.num_samples:
        ip = 0.
        for k in range(num_feature_blocks):
            ip_block = 2 * samplesf[k].reshape((-1, num_training_samples), order='F').T.dot(np.conj(new_sample[k].flatten(order='F')))
            ip += np.real(ip_block)
        gram_vector = ip
    elif num_training_samples > 0:
        ip = 0.
        for k in range(num_feature_blocks):
            ip_block = 2 * samplesf[k][:, :, :, :num_training_samples].reshape((-1, num_training_samples), order='F').T.dot(np.conj(new_sample[k].flatten(order='F')))
            ip += np.real(ip_block)
        gram_vector[:num_training_samples] = ip
    return gram_vector

def merge_samples(sample1, sample2, w1, w2, sample_merge_type):
    alpha1 = w1 / (w1 + w2)
    alpha2 = 1 - alpha1
    if sample_merge_type == 'replace':
        merged_sample = sample1
    elif sample_merge_type == 'merge':
        num_feature_blocks = len(sample1)
        merged_sample = [alpha1 * sample1[k] + alpha2 * sample2[k] for k in range(num_feature_blocks)]
    return merged_sample

def update_distance_matrix(distance_matrix, gram_matrix, gram_vector, new_sample_norm, id1, id2, w1, w2):
    """
        update the distance matrix
    """
    alpha1 = w1 / (w1 + w2)
    alpha2 = 1 - alpha1
    if id2 < 0:
        norm_id1 = gram_matrix[id1, id1]

        # udpate the gram matrix
        if alpha1 == 0:
            gram_matrix[:, id1] = gram_vector
            gram_matrix[id1, :] = gram_matrix[:, id1]
            gram_matrix[id1, id1] = new_sample_norm
        elif alpha2 == 0:
            # new sample is discard
            pass
        else:
            # new sample is merge with an existing sample
            gram_matrix[:, id1] = alpha1 * gram_matrix[:, id1] + alpha2 * gram_vector
            gram_matrix[id1, :] = gram_matrix[:, id1]
            gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector[id1]

        # udpate distance matrix
        distance_matrix[:, id1] = np.maximum(gram_matrix[id1, id1] + np.diag(gram_matrix) - 2 * gram_matrix[:, id1], 0)
        distance_matrix[:, id1][np.isnan(distance_matrix[:, id1])] = 0
        distance_matrix[id1, :] = distance_matrix[:, id1]
        distance_matrix[id1, id1] = np.Inf
    else:
        if alpha1 == 0 or alpha2 == 0:
            raise("Error!")

        norm_id1 = gram_matrix[id1, id1]
        norm_id2 = gram_matrix[id2, id2]
        ip_id1_id2 = gram_matrix[id1, id2]

        # handle the merge of existing samples
        gram_matrix[:, id1] = alpha1 * gram_matrix[:, id1] + alpha2 * gram_matrix[:, id2]
        distance_matrix[:, id1][np.isnan(distance_matrix[:, id1])] = 0
        gram_matrix[id1, :] = gram_matrix[:, id1]
        gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * norm_id2 + 2 * alpha1 * alpha2 * ip_id1_id2
        gram_vector[id1] = alpha1 * gram_vector[id1] + alpha2 * gram_vector[id2]

        # handle the new sample
        gram_matrix[:, id2] = gram_vector
        gram_matrix[id2, :] = gram_matrix[:, id2]
        gram_matrix[id2, id2] = new_sample_norm

        # update the distance matrix
        distance_matrix[:, id1] = np.maximum(gram_matrix[id1, id1] + np.diag(gram_matrix) - 2 * gram_matrix[:, id1], 0)
        distance_matrix[id1, :] = distance_matrix[:, id1]
        distance_matrix[id1, id1] = np.Inf
        distance_matrix[:, id2] = np.max(gram_matrix[id2, id2] + np.diag(gram_matrix) - 2 * gram_matrix[:, id2], 0)
        distance_matrix[id2, :] = distance_matrix[:, id2]
        distance_matrix[id2, id2] = np.Inf
    return distance_matrix, gram_matrix

def update_prior_weights(prior_weights, sample_weights, latest_ind, frame_num):
    # udpate the training sample weights
    if frame_num == 1:
        replace_ind = 1
        prior_weights[replace_ind] = 1
    else:
        if config.sample_replace_strategy == 'lowest_prior':
            replace_idx = np.argmin(prior_weights)
        elif config.sample_replace_strategy == 'lowest_weight':
            replace_idx = np.argmin(sample_weights)
        elif config.sample_replace_strategy == 'lowest_median_prior':
            median_prior = np.median(prior_weights)
        elif config.sample_replace_strategy == 'constant_tail':
            idx = np.sort(prior_weights)
            lt_idx = idx[1:config.lt_size]
            st_idx = idx[1:config.lt_size+1:]

            minw = np.min(prior_weights[st_idx])
            if minw != 0:
                lt_mask = np.zeros(prior_weights.shape, dtype=np.uint8)
                lt_mask[lt_idx] = True
                lt_mask = lt_mask & (prior_weights > 0)
                prior_weights[lt_mask] = minw * (1 - config.learning_rate)
        prior_weights = prior_weights / np.sum(prior_weights)
        return prior_weights, replace_idx

def update_sample_space_model(samplesf, new_train_sample, distance_matrix, gram_matrix,
        prior_weights, num_training_samples):
    """
    """
    num_feature_blocks = len(new_train_sample)

    # find the inner product of the new sample with existing samples
    gram_vector = find_gram_vector(samplesf, new_train_sample, num_training_samples)

    # find the inner product of the new sample with existing samples
    new_train_sample_norm = 0.

    for i in range(num_feature_blocks):
        new_train_sample_norm += np.real(2 * np.vdot(new_train_sample[i].flatten(), new_train_sample[i].flatten()))

    dist_vector = np.maximum(new_train_sample_norm + np.diag(gram_matrix) - 2 * gram_vector, 0)
    dist_vector[num_training_samples:] = np.Inf

    merged_sample = []
    new_sample = []
    merged_sample_id = -1
    new_sample_id = -1

    if num_training_samples == config.num_samples:
        min_sample_id = np.argmin(prior_weights)
        min_sample_weight = prior_weights[min_sample_id]

        if min_sample_weight < config.minimum_sample_weight:

            # udpate distance matrix and the gram matrix
            distance_matrix, gram_matrix = update_distance_matrix(distance_matrix,
                                                                  gram_matrix,
                                                                  gram_vector,
                                                                  new_train_sample_norm,
                                                                  min_sample_id, -1, 0, 1)

            # normalize the prior weights so that the new sample gets weight as the learning rate
            prior_weights[min_sample_id] = 0
            prior_weights = prior_weights * (1 - config.learning_rate) / np.sum(prior_weights)
            prior_weights[min_sample_id] = config.learning_rate

            # set the new sample and new sample position in the samplesf
            new_sample_id = min_sample_id
            new_sample = new_train_sample
        else:
            # merge the new sample or merge two of the existing samples and insert the new sample
            closest_sample_to_new_sample = np.argmin(dist_vector)
            new_sample_min_dist = dist_vector[closest_sample_to_new_sample]

            # find the closest pair amongst existing samples
            closest_existing_sample_idx = np.argmin(distance_matrix.flatten())
            closest_existing_sample_pair = np.unravel_index(closest_existing_sample_idx, distance_matrix.shape)
            existing_samples_min_dist = distance_matrix[closest_existing_sample_pair[0], closest_existing_sample_pair[1]]
            closest_existing_sample1, closest_existing_sample2 = closest_existing_sample_pair
            if closest_existing_sample1 == closest_existing_sample2:
                raise("Score matrix diagnoal filled wrongly")

            if new_sample_min_dist < existing_samples_min_dist:
                # renormalize prior weights
                prior_weights = prior_weights * (1 - config.learning_rate)

                # set the position of the merged sample
                merged_sample_id = closest_sample_to_new_sample

                # extract the existing sample the merge
                existing_sample_to_merge = []
                for i in range(num_feature_blocks):
                    existing_sample_to_merge.append(samplesf[i][:, :, :, merged_sample_id])

                # merge the new_training_sample with existing sample
                merged_sample = merge_samples(existing_sample_to_merge,
                                              new_train_sample,
                                              prior_weights[merged_sample_id],
                                              config.learning_rate,
                                              config.sample_merge_type)

                # update distance matrix and the gram matrix
                distance_matrix, gram_matrix = update_distance_matrix(distance_matrix,
                                                                      gram_matrix,
                                                                      gram_vector,
                                                                      new_train_sample_norm,
                                                                      merged_sample_id,
                                                                      -1,
                                                                      prior_weights[merged_sample_id],
                                                                      config.learning_rate)

                # udpate the prior weight of the merged sample
                prior_weights[closest_sample_to_new_sample] = prior_weights[closest_sample_to_new_sample] + config.learning_rate

            else:
                # merge the nearest existing samples and insert the new sample

                # renormalize prior weights
                prior_weights = prior_weights * ( 1 - config.learning_rate)

                if prior_weights[closest_existing_sample2] > prior_weights[closest_existing_sample1]:
                    tmp = closest_existing_sample1
                    closest_existing_sample1 = closest_existing_sample2
                    closest_existing_sample2 = tmp

                sample_to_merge1 = []
                sample_to_merge2 = []
                for i in range(num_feature_blocks):
                    sample_to_merge1.append(samplesf[i][:, :, :, closest_existing_sample1])
                    sample_to_merge2.append(samplesf[i][:, :, :, closest_existing_sample2])

                # merge the existing closest samples
                merged_sample = merge_samples(sample_to_merge1,
                                              sample_to_merge2,
                                              prior_weights[closest_existing_sample1],
                                              prior_weights[closest_existing_sample2],
                                              config.sample_merge_type)

                # update distance matrix and the gram matrix
                distance_matrix, gram_matrix = update_distance_matrix(distance_matrix,
                                                                      gram_matrix,
                                                                      gram_vector,
                                                                      new_train_sample_norm,
                                                                      closest_existing_sample1,
                                                                      closest_existing_sample2,
                                                                      prior_weights[closest_existing_sample1],
                                                                      prior_weights[closest_existing_sample2])

                # update prior weights for the merged sample and the new sample
                prior_weights[closest_existing_sample1] = prior_weights[closest_existing_sample1] + prior_weights[closest_existing_sample2]
                prior_weights[closest_existing_sample2] = config.learning_rate

                # set the mreged sample position and new sample position
                merged_sample_id = closest_existing_sample1
                new_sample_id = closest_existing_sample2

                new_sample = new_train_sample
    else:
        # if the memory is not full, insert the new sample in the next empty location
        sample_position = num_training_samples

        # update the distance matrix and the gram matrix
        distance_matrix, gram_matrix = update_distance_matrix(distance_matrix,
                                                              gram_matrix,
                                                              gram_vector,
                                                              new_train_sample_norm,
                                                              sample_position, -1, 0, 1)

        # update the prior weight
        if sample_position == 0:
            prior_weights[sample_position] = 1
        else:
            prior_weights = prior_weights * (1 - config.learning_rate)
            prior_weights[sample_position] = config.learning_rate

        new_sample_id = sample_position
        new_sample = new_train_sample

    if abs(1 - np.sum(prior_weights)) > 1e-5:
        raise("weights not properly udpated")

    return merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights
