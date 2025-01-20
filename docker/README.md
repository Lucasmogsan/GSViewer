```bash
cd docker
docker build -t gsviewer_image .
./run.sh

```

Locally run:
```bash
xhost +local:root
```

test in container with:
```bash
glxgears
```

Launch viewer
```bash
python main.py
```