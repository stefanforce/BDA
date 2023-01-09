import re as regex
import pyspark.sql.functions as PY_SPARK_FUNC
import pyspark.sql.types as PY_SPARK_TYPES

# Defining macros using regex for extracting data from xml
REGEX_REFERENCE_ARTICLE = regex.compile(r"Row\(_IdType='pubmed', _VALUE=`?\"?'?([0-9]+)`?\"?'?\)")
REGEX_AFFILIATION_ARTICLE = regex.compile(r"Row\(Affiliation=[\"'](.+?)[\"']")
REGEX_AGENCY_COUNTRY_ARTICLE = regex.compile(r", Agency=[\"'](.+?)[\"'], Country=")


# Parsing references from xml file
def parse_references_from_xml(input_references):
    if input_references is not None:
        output_dict = dict.fromkeys(regex.findall(REGEX_REFERENCE_ARTICLE, str(input_references)))
    else:
        return list()
    return list(output_dict)


# Parsing the authors tab in the xml file in order to find the total number of authors
# per article using affiliation tag
def parse_number_authors_using_affiliation_from_xml_affil(input_authors):
    if input_authors is not None:
        output_dict = [iterator_affiliation for iterator_affiliation in
                       list(dict.fromkeys
                            (regex.findall(REGEX_AFFILIATION_ARTICLE, str(input_authors))))
                       if len(iterator_affiliation) != 0
                       ]
    else:
        return 0
    return list(output_dict)


# Parsing the authors tab in the xml file in order to find the total number of authors
# per article using affiliation tag FOR SHOWING AFFILIATION PER ARTICLES

def parse_number_authors_using_affiliation_from_xml(input_authors):
    if input_authors is not None:
        output_dict = [iterator_affiliation for iterator_affiliation in
                       list(dict.fromkeys
                            (regex.findall(REGEX_AFFILIATION_ARTICLE, str(input_authors))))
                       if len(iterator_affiliation) != 0
                       ]
    else:
        return 0
    return len(output_dict)


# Parsing the grants tab in the xml file in order to find the total number of grants
# per article using agency tag
def parse_number_grants_agencies_using_agency_from_xml(input_grants):
    if input_grants is not None:
        output_dict = [iterator_agency for iterator_agency in
                       list(dict.fromkeys
                            (regex.findall(REGEX_AGENCY_COUNTRY_ARTICLE, str(input_grants))))
                       if len(iterator_agency) != 0
                       ]
    else:
        return 0
    return len(output_dict)


def adjust_integer_value(input_value):
    return 0 if input_value < 0 else input_value


# Creating UDF macros for specific types we need: references, grants and authors
REFERENCE_UDF = PY_SPARK_FUNC.udf(parse_references_from_xml,
                                  PY_SPARK_TYPES.ArrayType(PY_SPARK_TYPES.StringType(), False))

AUTHORS_UDF = PY_SPARK_FUNC.udf(parse_number_authors_using_affiliation_from_xml,
                                PY_SPARK_TYPES.IntegerType())

AUTHORS_AFFILIATION_UDF = PY_SPARK_FUNC.udf(parse_number_authors_using_affiliation_from_xml_affil,
                                            PY_SPARK_TYPES.ArrayType(PY_SPARK_TYPES.StringType(), False))

GRANTS_UDF = PY_SPARK_FUNC.udf(parse_number_grants_agencies_using_agency_from_xml,
                               PY_SPARK_TYPES.IntegerType())

INTEGER_ADJUST_UDF = PY_SPARK_FUNC.udf(adjust_integer_value, PY_SPARK_TYPES.IntegerType())


# Function which add new column into the Pyspark Dataframe
def add_custom_dataframe_column(input_dataframe, input_column_name, input_column_xml, input_lambda_function,
                                entry_value):
    # At first, we give the default values for a column, the name and default value
    input_dataframe = input_dataframe.withColumn(input_column_name, entry_value)

    # Now, we assign new value to our column using the specific function given as
    # lambda function as an input parameter
    # And use Pyspark function coalesce : https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.functions.coalesce.html
    # in order to erase the case when
    try:
        input_dataframe = input_dataframe.withColumn(input_column_name, input_lambda_function(input_column_xml))
        input_dataframe.withColumn(input_column_name,
                                   PY_SPARK_FUNC.coalesce(input_dataframe[input_column_name], entry_value))
    except:
        pass
    return input_dataframe


def transpose_xml_into_dataframe(sqlContext, input_path, trainModel = False):
    output_dataframe = sqlContext.read.format("xml") \
        .option("rootTag", "PubmedArticleSet") \
        .option("rowTag", "PubmedArticle") \
        .load(input_path)

    # Example of sample for PubmedID:
    #       <MedlineCitation Status="MEDLINE" Owner="NLM">
    #           <PMID Version="1">2486375</PMID>
    output_dataframe = add_custom_dataframe_column(output_dataframe, "PubmedID", "MedLineCitation.PMID._VALUE",
                                                   lambda x: PY_SPARK_FUNC.col(x).cast(PY_SPARK_TYPES.StringType()),
                                                   PY_SPARK_FUNC.lit(None))

    # Example of sample for Authors:<AuthorList CompleteYN="Y">
    #       <Article .....
    #           ...........
    #           <Author ValidYN="Y">
    #             <LastName>Lyon</LastName>
    #             <ForeName>R M</ForeName>
    #           ........
    #           </Author>
    #           <Author ValidYN="Y">
    #             <LastName>Woo</LastName>
    output_dataframe = add_custom_dataframe_column(output_dataframe, "NumberAuthors",
                                                   "MedLineCitation.Article.AuthorList.Author",
                                                   lambda x: PY_SPARK_FUNC.size(x), PY_SPARK_FUNC.lit(0))

    output_dataframe = add_custom_dataframe_column(output_dataframe, "NumberAffiliationAuthor",
                                                   "MedLineCitation.Article.AuthorList.Author",
                                                   lambda x: AUTHORS_UDF(PY_SPARK_FUNC.col(x)), PY_SPARK_FUNC.lit(0))

    output_dataframe = add_custom_dataframe_column(output_dataframe, "ListOfAffiliations",
                                                   "MedLineCitation.Article.AuthorList.Author",
                                                   lambda x: AUTHORS_AFFILIATION_UDF(PY_SPARK_FUNC.col(x)),
                                                   PY_SPARK_FUNC.lit(0))

    output_dataframe = add_custom_dataframe_column(output_dataframe, "TitleLength",
                                                   "MedLineCitation.Article.ArticleTitle",
                                                   lambda x: PY_SPARK_FUNC.length(x), PY_SPARK_FUNC.lit(0))

    # Example of sample for Keywords:
    #       <KeywordList Owner="NOTNLM">
    #         <Keyword MajorTopicYN="N">Ablation Efficacy</Keyword>
    #         <Keyword MajorTopicYN="N">Hard Tissue</Keyword>
    #         <Keyword MajorTopicYN="N">Nonlinear Optic</Keyword>
    #         <Keyword MajorTopicYN="N">Public Health</Keyword>
    #         <Keyword MajorTopicYN="N">Quantum Electronics</Keyword>
    #       </KeywordList>
    output_dataframe = add_custom_dataframe_column(output_dataframe, "NumberKeywords",
                                                   "MedLineCitation.KeywordList.Keyword",
                                                   lambda x: adjust_integer_value(PY_SPARK_FUNC.size(x)),
                                                   PY_SPARK_FUNC.lit(0))

    # Example same as above
    output_dataframe = add_custom_dataframe_column(output_dataframe, "NumberGrants",
                                                   "MedLineCitation.Article.GrantList.Grant",
                                                   lambda x: adjust_integer_value(PY_SPARK_FUNC.size(x)),
                                                   PY_SPARK_FUNC.lit(0))

    output_dataframe = add_custom_dataframe_column(output_dataframe, "NumberGrantAgencies",
                                                   "MedLineCitation.Article.GrantList",
                                                   lambda x: GRANTS_UDF(PY_SPARK_FUNC.col(x)), PY_SPARK_FUNC.lit(0))

    output_dataframe = add_custom_dataframe_column(output_dataframe, "References", "PubMedData.ReferenceList",
                                                   lambda x: REFERENCE_UDF(PY_SPARK_FUNC.col(x)), PY_SPARK_FUNC.array())

    # output_dataframe = add_custom_dataframe_column(output_dataframe, "Affiliation",
    #                                                "MedLineCitation.Article.AuthorList.Author.AffiliationInfo.Affiliation._VALUE",
    #                                                lambda x: AUTHORS_UDF(PY_SPARK_FUNC.col(x)), PY_SPARK_FUNC.lit(0))

    output_dataframe = output_dataframe.select(
        "PubMedID", "NumberAuthors", "NumberAffiliationAuthor", "ListOfAffiliations", "TitleLength",
        "NumberKeywords", "NumberGrants", "NumberGrantAgencies", "References"
    )

    # Modify the output_dataframe in order to get what we are looking for: a rating
    output_dataframe = output_dataframe.filter(PY_SPARK_FUNC.col("PubMedID").isNotNull())
    output_dataframe = output_dataframe.filter(PY_SPARK_FUNC.col("ListOfAffiliations").isNotNull())
    output_dataframe = output_dataframe.dropDuplicates(["PubMedID"])
    output_dataframe = output_dataframe.filter(output_dataframe.NumberAuthors > 0)
    output_dataframe = output_dataframe.withColumn("NumberReferences", PY_SPARK_FUNC.size("References"))

    # We create an auxiliary table where we add another column which will help us generated the rating
    #
    aux_dataframe_references = output_dataframe.select("PubMedID", "References", "ListOfAffiliations")
    aux_dataframe_references = aux_dataframe_references.withColumn("auxRefColumn",
                                                                   PY_SPARK_FUNC.arrays_zip("References")) \
        .withColumn("auxRefColumn", PY_SPARK_FUNC.explode("auxRefColumn")) \
        .withColumn("AffiliationList", PY_SPARK_FUNC.arrays_zip("ListOfAffiliations")) \
        .select(PY_SPARK_FUNC.col("auxRefColumn.References").alias("PubMedID_"),
                PY_SPARK_FUNC.col("PubMedID").alias('ReferencedBy'),
                PY_SPARK_FUNC.col("ListOfAffiliations").alias("AffiliationList"))

    aux_dataframe_references = aux_dataframe_references.groupBy("PubMedID_").agg(
        PY_SPARK_FUNC.count("ReferencedBy").alias("Rating"))

    output_dataframe = output_dataframe.join(aux_dataframe_references,
                                             output_dataframe.PubMedID == aux_dataframe_references.PubMedID_,
                                             "left").drop("PubMedID_")
    output_dataframe = output_dataframe.withColumn("Rating", PY_SPARK_FUNC.coalesce(aux_dataframe_references.Rating,
                                                                                    PY_SPARK_FUNC.lit(0)))
    output_dataframe = output_dataframe.sort(output_dataframe.Rating.desc())

    aux_dataframe_references = aux_dataframe_references.sort(aux_dataframe_references.Rating.desc())

    output_dataframe.show(600, truncate=False)
    output_dataframe.printSchema()

    aux_dataframe_references.show(600, truncate=False)
    aux_dataframe_references.printSchema()

    aux_dataframe_references_graph = output_dataframe.select("PubMedID")
    aux_dataframe_references_graph.write.option("header", True) \
        .csv("/IDs")

    if trainModel == True :
        output_dataframe = output_dataframe.drop("References")
        output_dataframe = output_dataframe.drop("ListOfAffiliations")


    return output_dataframe

def dataset_csv_to_dataframe(scContext, data_path):

    df_set = scContext.read.option("header",True) \
                .csv(data_path)
                
    df_set = df_set.withColumn("PubMedID",
                                      df_set["PubMedID"]
                                      .cast(PY_SPARK_TYPES.StringType()))

    df_set = df_set.withColumn("NumberAuthors",
                                        df_set["NumberAuthors"]
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("NumberAffiliationAuthor",
                                        df_set["NumberAffiliationAuthor"]
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("TitleLength",
                                        df_set["TitleLength"]   
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("NumberKeywords",
                                        df_set["NumberKeywords"]
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("NumberGrants",
                                        df_set["NumberGrants"]  
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("NumberGrantAgencies",
                                        df_set["NumberGrantAgencies"]
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("NumberReferences",
                                        df_set["NumberReferences"]  
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    df_set = df_set.withColumn("Rating",
                                        df_set["Rating"]
                                        .cast(PY_SPARK_TYPES.IntegerType()))

    return df_set
