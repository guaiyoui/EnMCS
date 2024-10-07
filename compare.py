import numpy as np


def e_step(error_rates, class_marginals, counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)

    patient_classes = np.zeros([nPatients, nClasses])
    # accelerate steps
    error_rates = np.repeat(error_rates.transpose(1, 0, 2)[np.newaxis, :, :, :], nPatients, axis = 0)
    counts = np.repeat(np.expand_dims(counts, axis = 1), nClasses, axis = 1)
    class_marginals = np.tile(class_marginals[np.newaxis, :, np.newaxis, np.newaxis],
                              (nPatients, 1, nObservers, nClasses))
    estimate = np.power(error_rates, counts) * class_marginals
    patient_classes = np.prod(estimate, axis = (-2, -1))
    patient_sum = np.sum(patient_classes, axis = 1)
    assert np.all(patient_sum > 0)
    patient_classes = patient_classes / patient_sum[:, np.newaxis]
    # print(patient_classes)

    return patient_classes

def e_step(error_rates, class_marginals, counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)

    patient_classes = np.zeros([nPatients, nClasses])

    for i in range(nPatients):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))

            patient_classes[i, j] = estimate
        # normalize error rates by dividing by the sum over all observation classes
        patient_sum = np.sum(patient_classes[i, :])
        if patient_sum > 0:
            patient_classes[i, :] = patient_classes[i, :] / float(patient_sum)

    return patient_classes