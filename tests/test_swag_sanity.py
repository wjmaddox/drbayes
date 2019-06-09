import torch

from torch import nn

from swag.posteriors import SWAG

def generate_dataloaders(N=100, batch_size=50):
    x = torch.rand(N, 1) * 12 - 6.
    y = 1.1 * x + torch.randn(N, 1) * 0.03
    dset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size)
    return loader


def model_generator():
    return torch.nn.Sequential(
        nn.Linear(1, 5),
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )


num_epochs = 100

subspaces = [
    {
        'name': 'covariance',
        'kwargs': {
            'max_rank': 20,
        }
    },
    {
        'name': 'pca',
        'kwargs': {
            'max_rank': 20,
            'pca_rank': 10,
        }
    },
    {
        'name': 'pca',
        'kwargs': {
            'max_rank': 20,
            'pca_rank': 'mle',
        }
    },
    {
        'name': 'freq_dir',
        'kwargs': {
            'max_rank': 20,
        }
    }
]

for item in subspaces:
    name, kwargs = item['name'], item['kwargs']
    print('Now running %s %r' % (name, kwargs))
    model = model_generator()
    small_swag_model = SWAG(base=model_generator,
                            subspace_type=name,
                            subspace_kwargs=kwargs)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    loader = generate_dataloaders(N=10)

    state_dict = None

    for epoch in range(num_epochs):
        model.train()
        
        for x, y in loader:
            model.zero_grad()
            pred = model(x)
            loss = ((pred - y)**2.0).sum()
            loss.backward()
            optimizer.step()
        small_swag_model.collect_model(model)

        if epoch == 4:
            state_dict = small_swag_model.state_dict()

    small_swag_model.fit()
    with torch.no_grad():
        x = torch.arange(-6., 6., 1.0).unsqueeze(1)
        for i in range(10):
            small_swag_model.sample(0.5)
            small_swag_model(x)

    _, _ = small_swag_model.get_space(export_cov_factor=False)
    _, _, _ = small_swag_model.get_space(export_cov_factor=True)
    small_swag_model.load_state_dict(state_dict)
