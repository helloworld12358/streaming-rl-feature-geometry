"""Fixed, non-learnable adapters for the complete current controller input.

The caller first constructs the exact vector the unmodified controller would
have received, including observation and bias.  An adapter may only map that
current vector; it never receives rewards, actions, latent labels, or future
samples.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ADAPTER_VERSION = "utilization-adapter-v2-complete-input"
ADAPTER_SCOPE = "complete_controller_input"
ADAPTER_NAMES = ("identity", "residual_rff", "tile_coding")
RFF_WIDTH = 64
RFF_FREQUENCY_SCALE = 1.0
TILE_NUM_TILINGS = 8
TILE_TABLE_SIZE = 512
TILE_TILES_PER_UNIT = 4.0


def derive_adapter_seed(environment: str, run_seed: int) -> int:
    """Derive a stable condition-independent seed without Python ``hash``."""

    payload = f"{ADAPTER_VERSION}\0{environment}\0{int(run_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def normalize_adapter_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate one adapter config and return its unique canonical form.

    Missing legacy configuration maps to the same canonical identity entry as
    an explicit identity configuration.  No invalid value is corrected.
    """

    raw = {"name": "identity"} if value is None else dict(value)
    name = raw.get("name")
    if name not in ADAPTER_NAMES:
        raise ValueError(f"utilization_adapter.name must be one of {list(ADAPTER_NAMES)}")

    if name == "identity":
        allowed = {"name"}
        canonical = {"name": name}
    elif name == "residual_rff":
        allowed = {"name", "width", "frequency_scale", "residual"}
        canonical = {
            "name": name,
            "width": int(raw.get("width", RFF_WIDTH)),
            "frequency_scale": float(raw.get("frequency_scale", RFF_FREQUENCY_SCALE)),
            "residual": raw.get("residual", True),
        }
        if canonical["width"] != RFF_WIDTH:
            raise ValueError(f"residual_rff width is fixed at {RFF_WIDTH}")
        if canonical["frequency_scale"] != RFF_FREQUENCY_SCALE:
            raise ValueError("residual_rff frequency_scale is fixed at 1.0")
        if canonical["residual"] is not True:
            raise ValueError("residual_rff must be residual")
    else:
        allowed = {
            "name",
            "num_tilings",
            "table_size",
            "tiles_per_unit",
            "residual",
        }
        canonical = {
            "name": name,
            "num_tilings": int(raw.get("num_tilings", TILE_NUM_TILINGS)),
            "table_size": int(raw.get("table_size", TILE_TABLE_SIZE)),
            "tiles_per_unit": float(raw.get("tiles_per_unit", TILE_TILES_PER_UNIT)),
            "residual": raw.get("residual", True),
        }
        if canonical["num_tilings"] != TILE_NUM_TILINGS:
            raise ValueError(f"tile_coding num_tilings is fixed at {TILE_NUM_TILINGS}")
        if canonical["table_size"] != TILE_TABLE_SIZE:
            raise ValueError(f"tile_coding table_size is fixed at {TILE_TABLE_SIZE}")
        if canonical["tiles_per_unit"] != TILE_TILES_PER_UNIT:
            raise ValueError("tile_coding tiles_per_unit is fixed at 4.0")
        if canonical["residual"] is not True:
            raise ValueError("tile_coding must be residual")

    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown utilization_adapter fields: {unknown}")
    return canonical


def _validate_input(controller_input: np.ndarray, input_dim: int) -> np.ndarray:
    if not isinstance(controller_input, np.ndarray):
        raise TypeError("utilization adapter input must be a numpy array")
    if controller_input.ndim != 1 or controller_input.shape != (input_dim,):
        raise ValueError(
            f"adapter expected input shape {(input_dim,)}, received {controller_input.shape}"
        )
    if controller_input.dtype != np.float64:
        raise TypeError("utilization adapter input must have dtype float64")
    if not np.isfinite(controller_input).all():
        raise FloatingPointError("utilization adapter received NaN or Inf")
    return controller_input


@dataclass(frozen=True)
class AdapterMetadata:
    version: str
    name: str
    scope: str
    input_dim: int
    block_dim: int
    output_dim: int
    adapter_seed: int | None
    residual: bool
    fixed_hyperparameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.version,
            "adapter_name": self.name,
            "adapter_scope": self.scope,
            "adapter_input_dim": self.input_dim,
            "adapter_block_dim": self.block_dim,
            "adapter_output_dim": self.output_dim,
            "adapter_seed": self.adapter_seed,
            "adapter_residual": self.residual,
            "adapter_fixed_hyperparameters": dict(self.fixed_hyperparameters),
        }


class UtilizationAdapter:
    """Base class with strict output validation and immutable metadata."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        environment: str,
        run_seed: int,
        input_dim: int,
        block_dim: int,
    ) -> None:
        if int(input_dim) < 1:
            raise ValueError("complete controller input dimension must be positive")
        self.config = normalize_adapter_config(config)
        self.environment = str(environment)
        self.run_seed = int(run_seed)
        self.input_dim = int(input_dim)
        self.block_dim = int(block_dim)
        self.output_dim = self.input_dim + self.block_dim

    def transform(self, controller_input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _validate_output(self, output: np.ndarray) -> np.ndarray:
        if not isinstance(output, np.ndarray):
            raise TypeError("utilization adapter output must be a numpy array")
        if output.shape != (self.output_dim,):
            raise ValueError(
                f"adapter expected output shape {(self.output_dim,)}, received {output.shape}"
            )
        if output.dtype != np.float64:
            raise TypeError("utilization adapter output must have dtype float64")
        if not np.isfinite(output).all():
            raise FloatingPointError("utilization adapter produced NaN or Inf")
        return output

    def metadata(self) -> AdapterMetadata:
        # Recording the deterministic derivation for every adapter keeps the
        # schema total; identity still owns no RNG and consumes no random draw.
        seed = derive_adapter_seed(self.environment, self.run_seed)
        fixed = {
            key: value
            for key, value in self.config.items()
            if key not in {"name", "residual"}
        }
        return AdapterMetadata(
            version=ADAPTER_VERSION,
            name=str(self.config["name"]),
            scope=ADAPTER_SCOPE,
            input_dim=self.input_dim,
            block_dim=self.block_dim,
            output_dim=self.output_dim,
            adapter_seed=seed,
            residual=bool(self.config.get("residual", False)),
            fixed_hyperparameters=fixed,
        )


class IdentityAdapter(UtilizationAdapter):
    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(config, block_dim=0, **kwargs)

    def transform(self, controller_input: np.ndarray) -> np.ndarray:
        controller_input = _validate_input(controller_input, self.input_dim)
        # Identity is intentionally a true no-op: no copy, cast, RNG, or state.
        return self._validate_output(controller_input)


class ResidualRFFAdapter(UtilizationAdapter):
    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(config, block_dim=RFF_WIDTH, **kwargs)
        rng = np.random.default_rng(derive_adapter_seed(self.environment, self.run_seed))
        self.W = rng.normal(
            0.0, RFF_FREQUENCY_SCALE, size=(RFF_WIDTH, self.input_dim)
        ).astype(np.float64, copy=False)
        self.b = rng.uniform(0.0, 2.0 * math.pi, size=RFF_WIDTH).astype(
            np.float64, copy=False
        )
        self.W.setflags(write=False)
        self.b.setflags(write=False)

    def transform(self, controller_input: np.ndarray) -> np.ndarray:
        controller_input = _validate_input(controller_input, self.input_dim)
        block = math.sqrt(2.0 / RFF_WIDTH) * np.cos(
            self.W @ controller_input + self.b
        )
        return self._validate_output(np.concatenate((controller_input, block)))


class TileCodingAdapter(UtilizationAdapter):
    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(config, block_dim=TILE_TABLE_SIZE, **kwargs)
        self._adapter_seed = derive_adapter_seed(self.environment, self.run_seed)
        tilings = np.arange(TILE_NUM_TILINGS, dtype=np.float64)[:, None]
        dimensions = np.arange(1, self.input_dim + 1, dtype=np.float64)[None, :]
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        self.offsets = np.mod(
            (tilings + 1.0) / TILE_NUM_TILINGS + dimensions * golden, 1.0
        )
        salts = []
        for index in range(self.input_dim):
            payload = f"{self._adapter_seed}\0tile-dimension\0{index}".encode("utf-8")
            salts.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
        self._salts = np.asarray(salts, dtype=np.uint64)
        self.offsets.setflags(write=False)
        self._salts.setflags(write=False)

    def tile_indices(self, controller_input: np.ndarray) -> np.ndarray:
        controller_input = _validate_input(controller_input, self.input_dim)
        coordinates = np.floor(
            controller_input[None, :] * TILE_TILES_PER_UNIT + self.offsets
        ).astype(np.int64)
        indices = np.empty(TILE_NUM_TILINGS, dtype=np.int64)
        mask = (1 << 64) - 1
        for tiling, row in enumerate(coordinates):
            mixed = np.bitwise_xor(
                row.view(np.uint64), self._salts
            )
            value = (self._adapter_seed ^ (tiling + 1)) & mask
            for item in mixed:
                value ^= int(item)
                value = (value * 1099511628211) & mask
            value ^= value >> 33
            value = (value * 0xFF51AFD7ED558CCD) & mask
            value ^= value >> 33
            indices[tiling] = value % TILE_TABLE_SIZE
        return indices

    def transform(self, controller_input: np.ndarray) -> np.ndarray:
        controller_input = _validate_input(controller_input, self.input_dim)
        block = np.zeros(TILE_TABLE_SIZE, dtype=np.float64)
        np.add.at(
            block,
            self.tile_indices(controller_input),
            1.0 / math.sqrt(TILE_NUM_TILINGS),
        )
        return self._validate_output(np.concatenate((controller_input, block)))


def make_utilization_adapter(
    config: Mapping[str, Any] | None,
    *,
    environment: str,
    run_seed: int,
    input_dim: int,
) -> UtilizationAdapter:
    canonical = normalize_adapter_config(config)
    kwargs = {
        "environment": environment,
        "run_seed": int(run_seed),
        "input_dim": int(input_dim),
    }
    if canonical["name"] == "identity":
        return IdentityAdapter(canonical, **kwargs)
    if canonical["name"] == "residual_rff":
        return ResidualRFFAdapter(canonical, **kwargs)
    return TileCodingAdapter(canonical, **kwargs)
