.. _llama2_7b_tp_ptl_lora_finetune_tutorial:

Fine-tuning Llama2 7B with tensor parallelism and LoRA using Neuron PyTorch-Lightning (``neuronx-distributed`` )
=========================================================================================

This tutorial shows how to fine-tune Llama2 7B with tensor parallelism and LoRA using Neuron PyTorch-Lightning APIs. For pre-training information, see the :ref:`Llama2 7B Tutorial <llama2_7b_tp_zero1_ptl_tutorial>`; 
for full fine-tuning information, see the :ref:`Fine-tuning Llama2 7B Tutorial <finetuning_llama2_7b_ptl_tutorial>`; For additional context, 
see the :ref:`Neuron PT-Lightning Developer Guide <ptl_developer_guide>`. 


Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^

For this experiment, we will use one trn1.32xlarge compute node in AWS EC2.
To set up the packages in the compute node, see
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.

Install the ``neuronx-distributed`` package inside the virtual environment using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Next, download the scripts for fine-tuning.


1. Create a directory to hold the experiments.

.. code:: ipython3

   mkdir -p ~/examples/tp_llama2_7b_hf_lora_finetune
   cd ~/examples/tp_llama2_7b_hf_lora_finetune

2. Download training scripts for the experiments.

.. code:: ipython3

   git clone https://github.com/aws-neuron/neuronx-distributed.git
   cd neuronx-distributed/examples/training/llama

3. Install the additional requirements and give the right permissions to the shell script.

.. code:: ipython3

   python3 -m pip install -r requirements.txt
   python3 -m pip install -r requirements_ptl.txt  # Currently we're supporting Lightning version 2.1.0
   chmod +x tp_zero1_llama2_7b_hf_finetune_ptl.sh

Download the Llama2-7B pre-trained checkpoint from HuggingFace.


1. Create a Python script ``get_model.py`` with the following lines: 

.. code:: ipython3

   import torch
   from transformers.models.llama.modeling_llama import LlamaForCausalLM
   model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
   torch.save(model.state_dict(), "llama-7b-hf-pretrained.pt")

2. Run the download script and conversion script to pull and convert the checkpoint, note that conversion scripts requires high memory:

.. code:: ipython3

   python3 get_model.py
   python3 convert_checkpoints.py --tp_size 8 --convert_from_full_model --config lightning/finetune_config/config.json --input_dir llama-7b-hf-pretrained.pt --output_dir llama7B-pretrained/
   cd lightning  # the folder of fine-tuning scripts using Neuron PyTorch-Lightning
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/test/integration/modules/lora/test_llama2_7b_lora_finetune.sh

Then, set the dataset for the fine-tuning job. In this example, we will use Dolly, which is an open source dataset
of instruction-following records on categories outlined in the InstructGPT paper, including brainstorming, classification,
closed QA, generation, information extraction, open QA, and summarization.

{
  "instruction": "Alice's parents have three daughters: Amy, Jessy, and what's the name of the third daughter?",
  
  "context": "",
  
  "response": "The name of the third daughter is Alice"
}


Configure the following flags in ``test_llama2_7b_lora_finetune.sh`` to set up the dataset:

.. code:: ipython3

   --data_dir "databricks/databricks-dolly-15k" \
   --task "open_qa" \


Before the actual fine-tune started, we need  to prepare the dataset

.. code:: ipython3

   python3 -c "import nltk; nltk.download('punkt')" 


In addition, you also need to enable LoRA with 

.. code:: ipython3

   --enable_lora \


The default example of LoRA configuration in ``tp_zero1_llama2_7b_hf_finetune_ptl.py`` is

.. code:: ipython3

   lora_config = LoraConfig(
        enable_lora=flags.enable_lora,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        lora_verbose=False,
        target_modules=["q_proj", "v_proj", "k_proj"],
   )


You can play with the these configurations as you like. At this point, you are all set to start fine-tuning.

Running fine-tuning
^^^^^^^^^^^^^^^^


By this step, the TRN node is all set up for running experiments. 

.. code:: ipython3

   bash tp_llama2_7b_lora_finetune.sh"

This script uses a tensor-parallel size of 8.
 
At the end of LoRA fine-tuning, the script will run evaluation once with a test data split by generating sentences and calculating ROUGE scores.
The final evaluation results and ROUGE score are then printed in your terminal.


LoRA Checkpoint
^^^^^^^^^^^^^^^^

There are three checkpoint saving modes for LoRA fine-tuning and you can set different modes by enabling different LoRA flags: 
``save_lora_base`` and ``merge_lora``.


* ``save_lora_base=False, merge_lora=False`` Save the LoRA adapter only.
* ``save_lora_base=True, merge_lora=False`` Save both the base model and the LoRA adapter seperately.
* ``save_lora_base=True, merge_lora=True`` Merge the LoRA adapter into the base model and then save the base model.


Other than the adapter, LoRA also needs to save the lora configuration file for adapter loading. 
You can save the configuration into the same checkpoint with the adapter, or save it as a seperately json file.
An example of LoRA flags for LoRA saving is

.. code:: ipython3

   lora_config = LoraConfig(
      save_lora_base=False,   # do not save the base model
      merge_lora=False,       # do not merge LoRA adapter into the base model
      save_lora_config_adapter,  # save LoRA checkpoint and configuration file in the same checkpoint
   )

After adding these flags, users can save LoRA model with 

.. code:: ipython3

   import neuronx_distributed as nxd
   nxd.save_checkpoint(checkpoint_dir_str="lora_adapter", tag="lora", model=model)


To enable checkpoint loading, add the following LoRA flags:

* ``load_lora_from_ckpt=True`` Resumes the checkpoint process.
* ``lora_save_dir="lora_adapter"`` Load LoRA checkpoint from the specified folder
* ``lora_load_tag="lora"`` Load the LoRA checkpoint with the specified tag

LoRA checkpoint will be loaded during LoRA initialization. 
Note that if LoRA configuration file is saved seperately, it should be placed as ``f{lora_save_dir}/adapter_config.json``.