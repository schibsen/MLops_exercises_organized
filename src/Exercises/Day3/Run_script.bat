set root=C:\Users\annas\anaconda3

call %root%\Scripts\activate.bat %root%

call conda list pandas

"C:\Users\annas\anaconda3\python.exe" "C:\Users\annas\OneDrive\Dokumente\DTU\8_semester\3weeks_MLops\MLops_exercises_organized\src\Exercises\Day3\vae_mnist_working.py"

python -m cProfile -o output_file.prof vae_mnist_working.py