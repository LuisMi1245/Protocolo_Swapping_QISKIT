#Corre el servidor Jupyterlab_Python dentro de la red LAN
#Ingresar con url-share-link token o password
#Password: 123456789

c:\Users\eulis\OneDrive\Desktop\QuantumComputing\venv\Scripts\Activate.ps1
cd C:\Users\eulis\OneDrive\Desktop\QuantumComputing\Jupyterlab\

$ip_addr = "$((Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias Wi-Fi).IPAddress)"
$dir="$((pwd).path)"
$env_path="$((Get-Command python.exe).path)"


Write-Host "Tu IP-LAN es: $ip_addr"
Write-Host "Te encuentras en: $dir"
Write-Host "Tu entorno virtual es: $env_path"
Write-Host "Ejecutando el servidor..."
jupyter-lab.exe --collaborative --ip=$ip_addr





#LINUX
#!/bin/sh

#Ingresar con url-share-link token o password
#password: 123456789

#source /home/spartan117/Escritorio/QuantumComputing/venv/bin/activate
#cd /home/spartan117/Escritorio/QuantumComputing/Jupyterlab

#ip_addr=`hostname -I | awk '{print $1}'`
#ip_host=`ipconfig.exe | grep -A4 Wi-Fi | grep IPv4 | cut -d":" -f 2 | tail -n1 | sed -e 's/\*//g'`
#dir=`pwd`
#env_path=`which python`

#echo "Creando proxy (WSL-VM y Windows)..."
#netsh.exe interface portproxy add v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress=$ip_addr
#echo "Tu IP WSL: $ip_addr"
#echo "Tu IP del host: $ip_host"
#echo "Te encuentras en: $dir"
#echo "Tu entorno virtual es: $env_path"
#echo "Ejecutando el servidor..."
#jupyter-lab --allow-root --collaborative --no-browser --ip=$ip_addr


#VOILA
#voila --Voila.ip=$ip_addr Optica.ipynb

