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
    def __init__(self, encoder, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0][:, 0, :])

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
