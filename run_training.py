from pipelines.training_pipeline import training_pipeline
from pipelines.promote_pipeline import promote_pipeline
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
def main():
    training_pipeline()
    promote_pipeline()

if __name__=='__main__':
    main()
