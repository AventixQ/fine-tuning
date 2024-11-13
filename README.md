# fine-tuning

### INSTRUKCJA ODPALANIA KODU NA KLASTRZE

1. Logowanie za pomocą SSH

ssh inf<indeks>@slurm.cs.put.poznan.pl

Podaj <hasło>

2. Nawiązanie sesji interaktywnej z dostępem do jednej z kart GPU

srun -p hgx --gres=gpu:1 -pty /bin/bash -l

3. Pobieranie anacondy

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

sh Anaconda3-2022.10-Linux-x86_64.sh

4. Tworzenie wirtualnego środowiska z conda

conda create -n PYTORCH

5. Ustawianie środowiska

- Aktywacja:

conda activate PYTORCH

- Instalacja PyTorcha (polecam wersje 12+):

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
-c pytorch -c nvidia

- Sprawdzenie czy torch działa odpowiednio:

python3

>>> import torch

>>> torch.cuda.is_available()

powinno zwrócić: True

>>> torch.version.cuda

powinno zwrócić wersje

6. Praca w środowisku z kodem

a. Wstaw niezbędne do pracy pliki:

- pobrany plik .xlsx (w kodzie nazwany 'examples-2000.xlsx' - pamiętaj, żeby dostosować nazwę)

- plik do fine-tuningu: fine-tuning-pp.py

- plik do instalacji wymagań: requirements.txt

Pliki możesz wstawiać z lokalnej konsoli m.in. za pomocą polecenia:

scp <ścieżka-do-pliku-lokalnie> inf<indeks>@slurm.cs.put.poznan.pl:~/.

(lub do innej ścieżki w katalogu zdalnym)

b. Zainstaluj niezbędne pakiety

- mając plik requirements.txt w środowisku, uruchom:

pip install -r requirements.txt

(jeżeli nie masz pip, polecam zainstalować)

c. Połącz się z Hugging Face jeżeli używasz modelu, który tego wymaga

- Wygeneruj token za stronie HF

- Sprawdź czy model, którego chcesz użyć jest dla ciebie dostępny (na część z nich trzeba zawnioskować)

- uruchom polecenie w celu autentykacji:

huggingface-cli login --token <YOUR_HUGGINGFACE_TOKEN>

d. stwórz w zdalnym środowisku foldery results i logs, sprawdź w kodzie czy wczytujesz pliki z odpowiedniej ścieżki

e. Stwórz skrypt w bashu, który uruchomi twój kod np. 'run_script.sh':

#!/bin/bash
conda run -n <NAZWA_ŚRODOWISKA> python3 fine-tuning-pp.py

Są inne wariacje tego pliku, ta jest najprostsza

f. Odpal kod np. za pomocą:

sbatch -p hgx -n1 -A inf<indeks> --gres=gpu:1 <skrypt-bash>.sh

g. Stan wykonania możesz sprawdzać np. za pomocą komendy:

squeue -u inf<indeks> -t all

h. Po wykonaniu zapytanie powstaje plik .out, który zawiera informacje o ewentualnych błędach, polecam go przeglądać.
 
