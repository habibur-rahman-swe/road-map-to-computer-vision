# Operating System Fundamentals

Understanding how operating systems manage resources is essential for working with computer vision applications, which are often resource-intensive.

---

## Files

### What is a File?
A file is a named collection of data stored on a persistent storage device (hard drive, SSD). It's the basic unit of data storage.

### File Components
- **Name**: Identifier for the file
- **Extension**: Indicates file type (e.g., `.txt`, `.py`, `.jpg`, `.mp4`)
- **Content**: The actual data (text, binary, image, video)
- **Metadata**: Size, creation date, permissions, location

### File Operations
- Create, read, write, delete, rename, copy, move
- File permissions (read, write, execute) control access

### File Paths
- **Absolute path**: Full path from root (e.g., `C:\Users\name\file.txt` or `/home/user/file.txt`)
- **Relative path**: Path relative to current directory (e.g., `./folder/file.txt`)

---

## Folders (Directories)

### What is a Folder?
A folder is a container that holds files and other folders. Folders organize files hierarchically.

### Directory Structure
- **Root directory**: Top-level directory (e.g., `C:\` on Windows, `/` on Unix)
- **Subdirectory**: A folder within another folder
- **Path**: The sequence of directories to reach a file

### Common Directory Conventions
- Project root contains source code, documentation, and configuration
- `src/` or `lib/` for source code
- `data/` for datasets
- `models/` for trained models
- `docs/` for documentation

---

## Paths

### Path Resolution
- **Absolute path**: Complete path from system root
- **Relative path**: Path from current working directory
- `.` refers to current directory, `..` refers to parent directory

### Path Examples
```
C:\Users\til-dev-pc-0\Desktop\road-map-to-computer-vision\
/home/user/projects/cv-roadmap/
./PHASE_0_COMPUTER_AND_LEARNING_FOUNDATIONS.md
../README.md
```

### Path Separators
- Windows: backslash `\`
- Unix/Linux/macOS: forward slash `/`
- Python's `os.path` and `pathlib` handle this automatically

---

## Processes

### What is a Process?
A process is a program in execution. It's an instance of a program running on the computer, with its own memory space and system resources.

### Process vs. Thread
- **Process**: Independent execution unit with separate memory
- **Thread**: Lightweight process sharing memory within a process

### Process Management
- **Creation**: Starting a new program
- **Scheduling**: CPU time allocation among processes
- **Communication**: Inter-process communication (pipes, sockets, shared memory)
- **Termination**: Ending a process

### Process States
- New, ready, running, waiting, terminated

---

## Memory

### What is Memory?
Memory is the storage area where data and programs reside while being executed.

### Memory Hierarchy
- **Registers**: Fastest, smallest, inside CPU
- **Cache**: Between CPU and RAM
- **RAM (Random Access Memory)**: Main memory, volatile (lost on power off)
- **Storage (SSD/HDD)**: Persistent, slower than RAM

### Memory Management
- **Allocation**: Assigning memory to processes
- **Deallocation**: Releasing memory when no longer needed
- **Virtual memory**: Extending RAM using disk space
- **Paging**: Dividing memory into fixed-size blocks
- **Segmentation**: Dividing memory into variable-size segments

### Memory in Python
- Automatic garbage collection
- References and reference counting
- Understanding memory leaks and optimization

---

## Programs

### What is a Program?
A program is a set of instructions that a computer executes to perform a task. It's the static form; when executed, it becomes a process.

### Program Execution
1. **Compilation**: Source code → machine code (C, C++, Rust)
2. **Interpretation**: Source code executed line-by-line (Python, JavaScript)
3. **Hybrid**: Combination (Java bytecode, .NET IL)

### Program Components
- **Code**: The instructions
- **Data**: Variables, constants, structures
- **Resources**: File handles, network connections, memory

### Running a Program
- Operating system loads program into memory
- Allocates resources
- Starts execution
- Monitors and manages until completion

---

## Key Relationships

```
Program (static code)
    ↓ (executed)
Process (running instance)
    ↓ (uses)
Memory (storage for code and data)
    ↓ (organized in)
Files & Folders (persistent storage)
    ↓ (accessed via)
Paths (location identifiers)
```

---

## Practical Implications for Computer Vision

- **Memory management**: CV applications process large images/videos; efficient memory use is critical
- **Processes**: Parallel processing for data augmentation or model inference
- **File paths**: Correctly handling dataset locations across operating systems
- **Program execution**: Understanding how frameworks like TensorFlow/PyTorch run on your system

---

## Recommended Learning Resources

- Operating system concepts (processes, memory, file systems)
- Command line proficiency
- Python's `os`, `sys`, and `pathlib` modules
- Basic system administration concepts

---

*This note provides foundational knowledge for understanding how computers execute programs and manage resources.*