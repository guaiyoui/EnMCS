@REM python accuracy_globalsearch.py -pt com -d acm || exit /b
@REM python accuracy_globalsearch.py -pt pri -d acm
@REM python accuracy_globalsearch.py -pt both -d acm
python accuracy_globalsearch.py -pt both_std -d acm
python accuracy_globalsearch.py -pt both_std_scores -d acm