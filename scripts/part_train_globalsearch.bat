@REM python main.py -d acm || exit /b
@REM python accuracy_globalsearch.py -d acm || exit /b
@REM python accuracy_globalsearch.py -pt com -d acm || exit /b
@REM python accuracy_globalsearch.py -pt pri -d acm
@REM python accuracy_globalsearch.py -pt both -d acm
@REM python accuracy_globalsearch.py -pt both_std -d acm
@REM python accuracy_globalsearch.py -pt both_std_scores -d acm

@REM python main.py -d dblp || exit /b
@REM python accuracy_globalsearch.py -d dblp
@REM python accuracy_globalsearch.py -pt com -d dblp
@REM python accuracy_globalsearch.py -pt pri -d dblp
@REM python accuracy_globalsearch.py -pt both -d dblp
@REM python accuracy_globalsearch.py -pt both_std -d dblp
@REM python accuracy_globalsearch.py -pt both_std_scores -d dblp

@REM python main.py -d freebase || exit /b
@REM python accuracy_globalsearch.py -d freebase
@REM python accuracy_globalsearch.py -pt com -d freebase
@REM python accuracy_globalsearch.py -pt pri -d freebase
@REM python accuracy_globalsearch.py -pt both -d freebase
@REM python accuracy_globalsearch.py -pt both_std -d freebase
@REM python accuracy_globalsearch.py -pt both_std_scores -d freebase

@REM python main.py -d imdb || exit /b
@REM python accuracy_globalsearch.py -d imdb
@REM python accuracy_globalsearch.py -pt com -d imdb
@REM python accuracy_globalsearch.py -pt pri -d imdb
@REM python accuracy_globalsearch.py -pt both -d imdb
@REM python accuracy_globalsearch.py -pt both_std -d imdb
@REM python accuracy_globalsearch.py -pt both_std_scores -d imdb

python main.py -d rm || exit /b
python accuracy_globalsearch.py -d rm
@REM python accuracy_globalsearch.py -pt com -d rm
@REM python accuracy_globalsearch.py -pt pri -d rm
@REM python accuracy_globalsearch.py -pt both -d rm
@REM python accuracy_globalsearch.py -pt both_std -d rm
@REM python accuracy_globalsearch.py -pt both_std_scores -d rm

python main.py -d terrorist || exit /b
python accuracy_globalsearch.py -d terrorist
@REM python accuracy_globalsearch.py -pt com -d terrorist
@REM python accuracy_globalsearch.py -pt pri -d terrorist
@REM python accuracy_globalsearch.py -pt both -d terrorist
@REM python accuracy_globalsearch.py -pt both_std -d terrorist
@REM python accuracy_globalsearch.py -pt both_std_scores -d terrorist

python main.py -d 3sources || exit /b
python accuracy_globalsearch.py -d 3sources

python main.py -d WikipediaArticles || exit /b
python accuracy_globalsearch.py -d WikipediaArticles