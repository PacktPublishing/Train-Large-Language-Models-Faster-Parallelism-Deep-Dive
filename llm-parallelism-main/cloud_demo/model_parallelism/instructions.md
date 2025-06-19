<!-- @format -->

sudo apt-get update
sudo apt-get install -y libaio-dev curl build-essential
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade deepspeed transformers datasets numpy
apt-get install nano

Follow the same steps as before on RunPod.io:
Upload: Save as setup_imdb_demo.sh and upload to /root/. by writing:
nano setup_complex_pipeline.sh
Save the File:
In nano, press Ctrl+O, then press Enter to save.
Press Ctrl+X to exit nano.

Make Executable:

chmod +x setup_imdb_demo.sh

Run:

./setup_imdb_demo.sh

cat setup_imdb_demo.sh
This should display the script to confirm it was saved correctly.
