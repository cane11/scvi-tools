import torch
from torch import nn as nn
from scvi.nn import FCLayers

class Hierarchical_Classifier(nn.Module):
    """
    Hierarchical Embedding Network
    Parameters (same as Classifier )
    ----------
    n_input
        Number of input dimensions (dimensions of the latent space)
    num_classes
        number of labels in each class in hierarchical list (ex : [2, 7])
    n_hidden
        Number of hidden nodes in one layer
    n_layers
        Number of hidden layers per NN (per independant representation)
    n_output
        Number of dimensions of each independant representation (not dependant on the number of labels)
    dropout_rate
        dropout_rate for nodes
    logits
        Return logits or not
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    activation_fn
        Valid activation function from torch.nn
    """

    def __init__(
        self,
        n_input: int,
        num_classes: list[int],
        n_output: int = 20,
        # modify the default of n_hidden?
        n_hidden: int = 128,
        logits: bool = False,
        dropout_rate: float = 0.1,
        activation_fn: nn.Module = nn.ReLU,
        n_layers: int = 1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.logits = logits
        self.n_input = n_input
        self.n_hidden = n_hidden

        # independant representation level 1 of root level
        layers_1 = [
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm = use_batch_norm,
                use_layer_norm= use_layer_norm,
                activation_fn=activation_fn,
            ),
            nn.Linear(n_hidden, n_output), 
        ]

        # independant representation level 2 of root level
        layers_2 = [
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm = use_batch_norm,
                use_layer_norm= use_layer_norm,
                activation_fn=activation_fn,
            ),
            nn.Linear(n_hidden, n_output), 
        ]
        # neural networks to obtain independant representations of dim n_output :
        self.lvl_2 = nn.Sequential(*layers_2)
        self.lvl_1 = nn.Sequential(*layers_1)

        if not logits : 
            self.output1 = nn.Sequential(
                nn.Linear(n_output, num_classes[0]), nn.Softmax(dim=-1)
            )
            self.output2 = nn.Sequential(
                nn.Linear(n_output + n_output, num_classes[1]), nn.Softmax(dim=-1)
            )

        else :
            self.output1= nn.Linear(n_output, num_classes[0]) 
            self.output2 = nn.Linear(n_output + n_output, num_classes[1]) 

    def forward(self, x):
        lvl_1_independant = self.lvl_1(x)
        lvl_2_independant = self.lvl_2(x)
        level_1 = self.output1(lvl_1_independant)
        # concatenation of independant representations for level 2
        level_2 = self.output2(torch.cat((lvl_1_independant, lvl_2_independant), dim=1))
        return level_1, level_2


class HierarchicalLossNetwork(Hierarchical_Classifier):
    """ "
    Parameters (same as Classifier )
    ----------
    n_input
        Number of input dimensions (dimensions of the latent space)
    hierarchical_labels
        labels organized in a hierarchical dictionary stored in hierarchy_dict.py
    num_classes
        number of labels in each class in hierarchical list (ex : [2, 7])
    n_hidden
        Number of hidden nodes in one layer
    n_layers
        Number of hidden layers per NN (per independant representation)
    n_output
        Number of dimensions of each independant representation
    dropout_rate
        dropout_rate for nodes
    logits
        Return logits or not
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    activation_fn
        Valid activation function from torch.nn
    level_one_labels, level_two_labels
        lists of labels from level 1 and level 2
    alpha 
        Parameter for layer loss
    beta 
        Parameter for dependance loss 
    """ 

    def __init__(
        self,
        n_input: int,
        hierarchical_labels : dict ,
        num_classes: list,
        n_output: int = 20,
        n_hidden: int = 128,
        n_layers: int = 1,
        logits: bool = False,
        dropout_rate: float = 0.1,
        activation_fn: nn.Module = nn.ReLU,
        device : str = "gpu",
        alpha=1,
        beta=0.8,
        p_loss=3,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            logits=logits,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            n_layers=n_layers,
            use_batch_norm = use_batch_norm,
            use_layer_norm= use_layer_norm,
            num_classes=num_classes,
        )

        self.total_level = len(num_classes)
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.hierarchical_labels = hierarchical_labels
        self.level_one_labels, self.level_two_lists = zip(*hierarchical_labels.items()) 
        self.level_two_labels = ()
        for l in self.level_two_lists :
            self.level_two_labels += tuple(l)
        self.numeric_hierarchy = self.words_to_indices()


    def words_to_indices(self):
        """
        Converts the classes from words to indices.
        """
        numeric_hierarchy = {}
        for k, v in self.hierarchical_labels.items(): 
            numeric_hierarchy[self.level_one_labels.index(k)] = [
                self.level_two_labels.index(i) for i in v
            ]
        return numeric_hierarchy

    def check_hierarchy(self, current_level, previous_level):
        """
        Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        """
        # check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [
            not current_level[i] in self.numeric_hierarchy[previous_level[i].item()]
            for i in range(previous_level.size()[0])
        ]

        return torch.FloatTensor(bool_tensor).to(self.device)

    def calculate_lloss(self, predictions, true_labels):
        """
        Calculates the layer loss (double Cross-Entropy)
        """

        lloss = 0
        for l in range(self.total_level):
            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])
        return self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):
        """
        Calculate the dependence loss, that enforces the hierarchy. 
        """

        dloss = 0
        for l in range(1, self.total_level):
            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)  
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l - 1]), dim=1) 
            #check whether the predictions enforce the hierarchy 
            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

            l_prev = torch.where(
                prev_lvl_pred == true_labels[l - 1],
                torch.FloatTensor([0]).to(self.device),
                torch.FloatTensor([1]).to(self.device),
            )
            l_curr = torch.where(
                current_lvl_pred == true_labels[l],
                torch.FloatTensor([0]).to(self.device),
                torch.FloatTensor([1]).to(self.device),
            )

            dloss += torch.sum(
                torch.pow(self.p_loss, D_l * l_prev)
                * torch.pow(self.p_loss, D_l * l_curr)
                - 1
            )

        return self.beta * dloss