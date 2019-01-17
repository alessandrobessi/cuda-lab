apt update -qq;
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb;
dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb;
apt-key add /var/cuda-repo-8-0-local-ga2/7fa2af80.pub;
apt-get update -qq;
apt install cuda-8.0;
apt-get install cuda gcc-5 g++-5 -y -qq;
ln -sf /usr/bin/gcc-5 /usr/local/cuda/bin/gcc;
ln -sf /usr/bin/g++-5 /usr/local/cuda/bin/g++;

/usr/local/cuda/bin/nvcc --version
gcc --version
g++ --version

echo "import os"
echo "os.environ['PATH'] += ':/usr/local/cuda/bin'"