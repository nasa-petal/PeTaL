"""
Version created on Thursday Feb 26 14:00:00 2019
@author: lauren e. friend lauren.e.friend@nasa.gov
@editor: courtney t. schiebout

*note: This program only runs on Python 3.7 or above
"""
from abstract_rank import clean_n_score
from highwire_scraper import get_articles_from_highwire_repository


def pull_from_highwire_and_rank(highwire_dir, stfp, journal_fp, article_pik_path, output_dir, cluster="n"):
    """
    This is the main function, it combines the functionality of highwire_scraper.py and "abstract_rank.py.
    Parameters
    ----------
    highwire_dir: file path
        maps to the folder containing the zip files downloaded from highwire
    stfp: file path
        maps to the json file containing the engineering to biology thesaurus
    journal_fp: file path
        maps to the text file containing the abbreviations for the desired journals
    article_pik_path: file path
        maps to the folder that the pickled items should be saved
    output_dir: file path
        maps to the folder that all of the engineer_top50.txt files are written to
    cluster: str
        "y" outputs cluster compatible files, "n" outputs petal compatible files.

    Returns
    -------
        pickle of Articles and ScoredArticles
        engineer.txt top 50 files
    """

    articles = get_articles_from_highwire_repository(journal_fp, highwire_dir, article_pik_path)  # get list of Articles
    clean_n_score(stfp, articles, article_pik_path, output_dir, cluster)  # return top 50 Articles for each eng. term
    print(f"Journal articles have been scored and ranked! find them in {output_dir}")


if __name__ == "__main__":
    """
    Below, the user inputs the HighWire directory, the file path for the search terms that are utilized 'stfp',
           and the file path the user desires for the ranked abstracts.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw_dir", default="C:/Users/cschiebo/Envs/PeTaL/petal/scrape-n-rank/HighWire",
                        help="The file path for the compressed HighWire archives")
    parser.add_argument("--stfp", default="NTRS.js",
                        help="file path for the engineering thesaurus JSON file")
    parser.add_argument("--output_dir", default="data", help="file path for the final ranked text files")
    parser.add_argument(
        "--article_output_path", default="animal_articles",
        help="intermediate data storage file path")
    parser.add_argument(
        "--jrnl_abbreviations_fp", default="journal_inputs.txt",
        help="file path for the desired journal abbreviation text file")
    parser.add_argument(
        "--cluster", default="n",
        help="specify 'y' to have file output in cluster compatible format")
    args = parser.parse_args()
    pull_from_highwire_and_rank(
        args.hw_dir, args.stfp, args.jrnl_abbreviations_fp, args.article_output_path, args.output_dir, args.cluster
    )
    print("the process is complete")
