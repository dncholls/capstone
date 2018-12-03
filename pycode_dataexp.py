from pandas import * # [6]
from matplotlib.pyplot import * # [4]
from numpy import *	 # [7]
from pymysql import * # [8]
from scipy.stats import kruskal, chisquare # [5]
from seaborn import * # [9]

## ~~~~~~ Connecting AWS to Python ~~~~~~ ##
host="[host here]"
port="[port here]"
dbname="[dbname here]"
user="[username here]"
password="[password here]"

mrdata = connect(host, user=user, port=port, passwd=password, db=dbname)
# [3]

## ~~~~~~ Data Cleaning Functions ~~~~~~ ##
def period_to_underscore(the_input):
	the_input.columns = [i.replace(".", "_") for i in the_input.columns]

	return the_input

def datetime_converter(the_input, file):
	
	the_input['DateTime'] = to_datetime(the_input['DateTime'], format = "%m/%d/%Y %I:%M:%S %p")
	the_input['MonthYear'] = to_datetime(the_input['MonthYear'], format = "%m/%d/%Y %I:%M:%S %p")
	the_input['MonthYear'] = the_input.MonthYear.values.astype('datetime64[M]')

	if file == 1:
		the_input['Date_of_Birth'] = to_datetime(the_input['Date_of_Birth'], format = "%m/%d/%Y")
		the_input['Date_of_Birth'] = the_input['Date_of_Birth'].values.astype("datetime64[D]")

	return the_input

def age_cleaning(the_input, file):
	if file == 1:
		week_cond = the_input.Age_upon_Outcome.str.contains("week+")
		the_input.loc[week_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[week_cond].replace("(week)+s*", "", regex = True))*0.01917811

		day_cond = the_input.Age_upon_Outcome.str.contains("day+", na = False)
		the_input.loc[day_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[day_cond].replace("(day)+s*", "", regex = True))*0.00273973

		month_cond = the_input.Age_upon_Outcome.str.contains("month+", na = False)
		the_input.loc[month_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[month_cond].replace("(month)+s*", "", regex = True))*0.083333545491

		year_cond = the_input.Age_upon_Outcome.str.contains("year+", na = False)
		the_input.loc[year_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[year_cond].replace("(year)+s*", "", regex = True))

		the_input.Age_upon_Outcome = to_numeric(the_input.Age_upon_Outcome)

	else:
		week_cond = the_input.Age_upon_Intake.str.contains("week+")
		the_input.loc[week_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[week_cond].replace("(week)+s*", "", regex = True))*0.01917811

		day_cond = the_input.Age_upon_Intake.str.contains("day+", na = False)
		the_input.loc[day_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[day_cond].replace("(day)+s*", "", regex = True))*0.00273973

		month_cond = the_input.Age_upon_Intake.str.contains("month+", na = False)
		the_input.loc[month_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[month_cond].replace("(month)+s*", "", regex = True))*0.083333545491

		year_cond = the_input.Age_upon_Intake.str.contains("year+", na = False)
		the_input.loc[year_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[year_cond].replace("(year)+s*", "", regex = True))

		the_input.Age_upon_Intake = to_numeric(the_input.Age_upon_Intake)

	return the_input

def unknown_sex(the_input, file):
	if file == 1:
		the_input.loc[the_input.Sex_upon_Outcome.isna(),"Sex_upon_Outcome"] = "Unknown"

	else:
		the_input.loc[the_input.Sex_upon_Intake.isna(), "Sex_upon_Intake"] = "Unknown"

	return the_input

def total_clean(the_input, file):
	the_input = period_to_underscore(the_input)
	the_input = datetime_converter(the_input, file)
	the_input = age_cleaning(the_input, file)
	the_input = unknown_sex(the_input, file)

	return the_input

# [6], [7]

## ~~~~~~ Importing and Cleaning the Data ~~~~~~ ##
## When Importing Data, the following rows are removed:
### wherever Outcome.Type is <NA>.
### wherever Age.upon.Outcome is <NA>.
### Total Loss of Rows: 13.

# Data Import.
dog_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Dog" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
cat_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Cat" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
bird_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Bird" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
total_outcomes = read_sql('select `Age.upon.Outcome`, `Outcome.Type` from raw_outcomes where `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con = mrdata)
all_data_out = read_sql('select * from raw_outcomes where `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con = mrdata)
all_data_in = read_sql('select * from raw_intakes;', con = mrdata) 
# [8]

# Cleaning the Data.
dog_outcomes = total_clean(dog_outcomes, 1)
cat_outcomes = total_clean(cat_outcomes, 1)
bird_outcomes = total_clean(bird_outcomes, 1)
total_outcomes = period_to_underscore(total_outcomes)
all_data_out = total_clean(all_data_out, 1)
all_data_in = total_clean(all_data_in, 0)

## ~~~~~~ Initial Data Exploration ~~~~~~ ##
# Printing Summary Statistics.
print("Summary Statistics for Intakes Dataset:")
print(all_data_in.describe(include = "all")) # [6]
print("")
print("Summary Statistics for Outcomes Dataset:")
print(all_data_out.describe(include = "all")) # [6]

# Preparing the Data for Plotting.
## I am Plotting Frequency vs Outcome Types for All Animals.
outcome_types = DataFrame()
outcome_types_2 = DataFrame()
total_outcome = DataFrame()
temp = DataFrame()

outcome_types['Outcome'] = dog_outcomes.loc[:,"Outcome_Type"]
outcome_types['Animal'] = "Dog"

temp['Outcome'] = cat_outcomes.loc[:,"Outcome_Type"]
temp['Animal'] = "Cat"

outcome_types = outcome_types.append(temp)

outcome_types_2['Outcome'] = bird_outcomes.loc[:,"Outcome_Type"]
outcome_types_2['Animal'] = "Bird"

total_outcome['Outcome'] = total_outcomes.loc[:,"Outcome_Type"]
# [6]

# Plotting the Outcome Types.
subplot(2,1,1)
countplot(x = "Outcome", hue = "Animal", data = outcome_types, palette = "Set2")
ylabel("Frequency")
xlabel("")
title("Frequency of Outcome Types for Dogs, Cats and Birds")
legend(loc = 1)

subplot(2,1,2)
countplot(x = "Outcome", hue = "Animal", data = outcome_types_2, palette = "Set2")
xlabel("Outcome Type")
ylabel("Frequency")
legend(loc = 1)
show()

countplot(x = "Outcome", data = total_outcome, palette = "Set2")
xlabel("Outcome Type")
ylabel("Frequency")
title("Frequency of Outcome Types for Dogs, Cats and Birds")
show()
# [9]

### Works Cited
# [1] Austin Animal Center. Austin Animal Center Intakes. October 1, 2018. Distributed by the official city of Austin open data portal. https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm.
# [2] Austin Animal Center. Austin Animal Center Outcomes. October 1, 2018. Distributed by the official city of Austin open data portal. https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238.
# [3] Berhane, Fiseeha. “Using Amazon Relational Database Service with Python and R.” datascience-enthusiast. Accessed September 23, 2017. https://datascience-enthusiast.com/R/AWS_RDS_R_Python.html.
# [4] Hunter, John D. "Matplotlib: A 2D graphics environment." Computing in science & engineering 9, no. 3 (2007): 90-95.
# [5] Jones, Eric, Travis Oliphant, and Pearu Peterson. "{SciPy}: open source scientific tools for {Python}." (2014).
# [6] McKinney, Wes. "Data structures for statistical computing in python." In Proceedings of the 9th Python in Science Conference, vol. 445, pp. 51-56. 2010.
# [7] Oliphant, Travis E. A guide to NumPy. Vol. 1. USA: Trelgol Publishing, 2006.
# [8] “PyMySQL Documentation.” Readthedocs. Accessed October 19, 2018. https://pymysql.readthedocs.io/en/latest/.
# [9] Waskom, Michael, Olga Botvinnik, Drew O'Kane, Paul Hobson, Joel Ostblom, Saulius Lukauskas, David C Gemperline, et al. “Mwaskom/seaborn: V0.9.0 (July 2018)”. Zenodo, July 16, 2018. doi:10.5281/zenodo.1313201.
