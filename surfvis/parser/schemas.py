from dataclasses import dataclass
import os.path
import glob
from typing import *
from scabha import configuratt
from scabha.cargo import Parameter
from omegaconf.omegaconf import OmegaConf


schema = None

@dataclass
class _CabInputsOutputs(object):
    inputs: Dict[str, Parameter]
    outputs: Dict[str, Parameter]

# load schema files
if schema is None:

    # all *.yaml files under pfb.parser will be loaded automatically

    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml"))

    structured = OmegaConf.structured(_CabInputsOutputs)

    schema = OmegaConf.create(configuratt.load_nested(files,
                                                      structured=structured,
                                                      config_class="SvisCleanCabs"))




