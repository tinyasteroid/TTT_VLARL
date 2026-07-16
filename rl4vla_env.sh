conda create -n rlvla_env3  python==3.10
conda activate rlvla_env3

conda deactivate
conda remove -n rlvla_env3 --all -y

pip install torch==2.2.0 
pip install numpy==1.26.4

#pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#pip install flash-attn==2.7.4.post1 --no-build-isolation   #if the above does not work, try this one.

# install other dependencies
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..







cd VLAC && pip install -e . && cd ..

#openvla 0.0.3 requires peft==0.11.1, but you have peft 0.15.2 which is incompatible.
#openvla 0.0.3 requires tokenizers==0.19.1, but you have tokenizers 0.21.4 which is incompatible.
#openvla 0.0.3 requires transformers==4.40.1, but you have transformers 4.51.3 which is incompatible.
#others
#pip install tokenizers==0.19.1
pip install tokenizers==0.21.4

#for video flash
pip install av==15.1.0
pip install decord==0.6.0




##############=no need to install#########
#############followings are for data collection and evaluation, which are not necessary for training the VLA model. You can skip them if you only want to train the VLA model.#############

#for nora
pip install qwen-vl-utils==0.0.14

#for tracevla
cd SimplerEnv/simpler_env/policies/tracevla/co-tracker && pip install -e . && cd ../../../../..
git clone https://github.com/umd-huang-lab/tracevla.git




cuda="0"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 2000 \
  --eval_steps 50 \
  --save_steps "0, 2000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --unnorm_key="bridge_orig" \
  --wandb_project "RLVLA_sft"



# following is for collecting data.
conda create -n octo_env -y python=3.10
conda activate octo_env

git clone https://github.com/octo-models/octo.git

cd ManiSkill && pip install -e . && cd ..

cd octo && pip install -e . && pip install -r requirements.txt && cd ..
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "nvidia-cudnn-cu11>=8.7,<9.0" --index-url https://download.pytorch.org/whl/cu118
pip install -U tyro
pip install scipy==1.12.0

cd SimplerEnv && pip install -e . && cd ..