import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from inception_time_pytorch.modules import InceptionModel


class InceptionTime():
    
    def __init__(self,
                 x,
                 y,
                 filters,
                 depth,
                 models):
        
        '''
        Implementation of InceptionTime model introduced in Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier,
        C., Schmidt, D.F., Weber, J., Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. Inceptiontime:
        Finding alexnet for time series classification. Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        y: np.array.
            Class labels, array with shape (samples,) where samples is the number of time series.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.
        
        models: int.
            The number of models.
        '''
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Calculate the scaling parameters.
        mu = np.nanmean(x, axis=0, keepdims=True)
        sigma = np.nanstd(x, axis=0, keepdims=True)

        # Calculate the length of the time series.
        input_size = x.shape[1]
        
        # Define the label mappings.
        self.id2label = {idx: label for idx, label in enumerate(np.sort(np.unique(y)))}
        self.label2id = {label: idx for idx, label in enumerate(np.sort(np.unique(y)))}

        # Encode the labels.
        y = np.array([self.label2id[i] for i in y])
        
        # Calculate the number of classes.
        num_classes = len(np.unique(y))
        
        # Calculate the class weights.
        self.weight = compute_class_weight(class_weight="balanced", classes=np.sort(np.unique(y)), y=y)
        
        # Save the data.
        self.x = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)
        
        # Build and save the models.
        self.models = [
            InceptionModel(
                input_size=input_size,
                num_classes=num_classes,
                filters=filters,
                depth=depth,
                mu=mu,
                sigma=sigma,
                seed=seed,
            ).to(self.device) for seed in range(models)
        ]
    
    def fit(self,
            learning_rate,
            batch_size,
            epochs,
            verbose=True):
        
        '''
        Train the models.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # Generate the training dataset.
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True
        )
        
        for m in range(len(self.models)):
            
            # Define the optimizer.
            optimizer = torch.optim.Adam(self.models[m].parameters(), lr=learning_rate)
            
            # Define the loss function.
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(self.weight).float().to(self.device))
            
            # Train the model.
            print(f'Training model {m + 1} on {self.device}.')
            self.models[m].train(True)
            for epoch in range(epochs):
                for features, target in dataset:
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    loss = loss_fn(output, target.to(self.device))
                    loss.backward()
                    optimizer.step()
                    accuracy = (torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1) == target).float().sum() / target.shape[0]
                if verbose:
                    print('epoch: {}, loss: {:,.6f}, accuracy: {:.6f}'.format(1 + epoch, loss, accuracy))
            self.models[m].train(False)
            print('-----------------------------------------')
            
    
    def predict(self, x):
        
        '''
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        Returns:
        __________________________________
        y: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        '''
    
        # Get the predicted probabilities.
        with torch.no_grad():
            p = torch.concat([torch.nn.functional.softmax(model(torch.from_numpy(x).float().to(self.device).float()), dim=-1).unsqueeze(-1) for model in self.models], dim=-1)
        
        # Get the predicted labels.
        y = np.array([self.id2label[int(i)] for i in p.mean(-1).argmax(-1).detach().cpu().numpy().flatten()])

        return y
