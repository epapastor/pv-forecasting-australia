import math
import torch
import torch.nn as nn


# ======================================================
# Positional Encoding
# ======================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # create the positional matrix: create a unique vector for each time step. 
        pe = torch.zeros(max_len, d_model)
        # Each row: timestep index
        position = torch.arange(0, max_len).unsqueeze(1)
        #Frequency scaling: Different sine/cosine frequencies. Low dimmensions-low oscillations/ high dimensios -> fast oscillations ( )
        # The model can infer relative distance
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        #Fill even/odd dimensions. So each gets a smooth and unique signature
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #Register as buffer to move 
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


# ======================================================
# Transformer Forecast Model
# ======================================================
class TransformerForecast(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        output_window,
    ):
        super().__init__()

        # -------- Input projection --------
        #from raw features to embeddings: we mix features and maps them into a learned space.. 
        self.input_proj = nn.Linear(input_size, d_model)

        # -------- Positional encoding --------
        # So now each timestep vector contains feature info and temporal postion
        self.pos_encoder = PositionalEncoding(d_model)

        # -------- Transformer encoder --------
        # Each layer as multi head self attention and FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # IMPORTANT
            activation="gelu",
        )
        # This applies the attention + FFN block repeatedly
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # -------- Prediction head --------
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_window),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, output_window)
        """

        x = self.input_proj(x)          # (B, L, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)         # (B, L, d_model)

        x = x[:, -1, :]                 # last timestep
        return self.head(x)
