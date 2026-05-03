import torch


def pad_collate_fn(batch):
    """
    Pads audio waveforms in a batch to the length of the longest waveform.
    Uses only the first two elements of each item: (waveform, label).
    Any extra elements in the batch items are ignored.
    """
    waveforms = []
    labels = []
    for item in batch:
        waveform, label = item[0], item[1]  # ignore anything beyond (waveform, label)
        waveforms.append(waveform)
        labels.append(torch.as_tensor(label))

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    labels = torch.stack(labels)
    return padded_waveforms, labels


def pad_collate_fn_aug(batch):
    """
    Collate function for the DatasetWithAugmentation. It handles batches of
    (original_waveform, augmented_waveform, label) tuples.
    """
    originals, augmenteds, labels = zip(*batch)
    padded_originals = torch.nn.utils.rnn.pad_sequence(
        list(originals), batch_first=True, padding_value=0.0
    )
    padded_augmenteds = torch.nn.utils.rnn.pad_sequence(
        list(augmenteds), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))
    return padded_originals, padded_augmenteds, labels


def pad_collate_fn_speaker(batch):
    """
    Pads audio waveforms and handles speaker IDs.
    Assumes each item in the batch is a (waveform, label, speaker_id) tuple.
    """
    # Unpack three items now
    waveforms, labels, speakers = zip(*batch)

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))

    # Speakers are usually strings, so we return them as a tuple
    return padded_waveforms, labels, speakers


def pad_collate_fn_speaker_source(batch):
    """
    Pads audio waveforms and returns speaker + source strings.
    Assumes each item is (waveform, label, speaker, source).
    """
    waveforms, labels, speakers, sources = zip(*batch)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))
    # speakers/sources are strings; return as tuples (kept as-is for your encoder)
    return padded_waveforms, labels, speakers, sources


def pad_collate_fn_speaker_source_multiclass(batch):
    """
    Pads audio waveforms and returns:
      waveforms, binary_labels, multiclass_labels, speakers, sources
    Assumes each item is:
      (waveform, binary_label, multi_label, speaker, audio_name)
    """
    waveforms, bin_labels, attack_id, speakers, sources = zip(*batch)

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    bin_labels = torch.stack(list(bin_labels))
    attack_id = torch.stack(list(attack_id))

    return padded_waveforms, bin_labels, attack_id, speakers, sources
