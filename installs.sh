# TODO make docker file
sudo apt update
sudo apt install -y libsdl2-dev ffmpeg tmux htop zip
aws s3 cp s3://gb-exp/PokemonRed.gb ./
