# casim
Computer Architecture Simulation Infrastructure for CSCE 614 Computer Architecture



##### 1. Apply a path file for cse server

```
$ git apply cse_server.patch
```

##### 2. Unzip benchmarks files

```
zip -F benchmarks.zip --out single-benchmark.zip && unzip single-benchmark.zip
```

### 3. Environemnt setup

Everytime you want to build or run zsim, you need to setup the environment variables first.

```
$ source setup_env
```

##### 4. Compile zsim

```
$ cd zsim
$ scons -j4
```

##### 5. Launch a test to run

```
./build/opt/zsim tests/simple.cfg
```



###### For more information, check `zsim/README.md`
