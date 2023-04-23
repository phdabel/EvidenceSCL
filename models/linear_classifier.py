import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    This is a linear classifier that takes the output of a pretrained model and
    adds a linear layer on top of it. The pretrained model can be any model
    from the transformers library (https://huggingface.co/transformers/pretrained_models.html).

    Args:
        encoder: A pretrained model from the transformers library.
        num_classes: The number of classes for the classifier.

    """
    def __init__(self, encoder, num_classes=3):
        super(LinearClassifier, self).__init__()
        # get weights of the last layer of the encoder
        dim_mlp = encoder.config.hidden_size
        self.num_classes = num_classes
        self.fc = nn.Linear(dim_mlp, num_classes)
        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """

        Args:
            input_ids: pair-level input ids
            attention_mask: pair-level attention mask
            labels: pair-level labels

        Returns:
            logits: pair-level logits for each class (if labels is not None)
            outputs: outputs of the encoder (if labels is None)

        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_outputs = outputs[1]
        sequence_outputs = self.dropout(sequence_outputs)
        logits = self.fc(sequence_outputs)
        # logits = self.fc(outputs[0][:, 0, :])
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
