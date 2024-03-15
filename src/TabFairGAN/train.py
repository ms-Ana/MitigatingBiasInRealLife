import torch
from preprocess import *
from Generator import Generator
from Critic import Critic
from FairLoss import *


def train(
    df,
    epochs: int = 500,
    batch_size: int = 64,
    fair_epochs: int = 10,
    lamda: float = 0.5,
    command: str = "with_fairness",
    device: str = "cuda:0",
    **kwargs
):
    if command == "with_fairness":
        (
            ohe,
            scaler,
            input_dim,
            discrete_columns,
            continuous_columns,
            train_dl,
            data_train,
            data_test,
            S_start_index,
            Y_start_index,
            underpriv_index,
            priv_index,
            undesire_index,
            desire_index,
        ) = prepare_data_with_fairness(df, batch_size, **kwargs)
    elif command == "no_fairness":
        (
            ohe,
            scaler,
            input_dim,
            discrete_columns,
            continuous_columns,
            train_dl,
            data_train,
            data_test,
        ) = prepare_data_no_fairness(df, batch_size)

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    if command == "with_fairness":
        second_critic = FairLossFunc(
            S_start_index,
            Y_start_index,
            underpriv_index,
            priv_index,
            undesire_index,
            desire_index,
        ).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    gen_optimizer_fair = torch.optim.Adam(
        generator.parameters(), lr=0.0001, betas=(0.5, 0.999)
    )
    crit_optimizer = torch.optim.Adam(
        critic.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    # loss = nn.BCELoss()
    critic_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        if i + 1 <= (epochs - fair_epochs):
            print("training for accuracy")
        if i + 1 > (epochs - fair_epochs):
            print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(device)
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(
                    size=(batch_size, input_dim), device=device
                ).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(
                    batch_size, input_dim, device=device, requires_grad=True
                )
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                crit_loss = get_crit_loss(
                    crit_fake_pred, crit_real_pred, gp, c_lambda=10
                )

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            #############################
            if i + 1 <= (epochs - fair_epochs):
                # training the generator for accuracy
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(
                    size=(batch_size, input_dim), device=device
                ).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)

                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()

                # Update the weights
                gen_optimizer.step()

            ###############################
            if i + 1 > (epochs - fair_epochs):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(
                    size=(batch_size, input_dim), device=device
                ).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                gen_fair_loss.backward()
                gen_optimizer_fair.step()
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim
