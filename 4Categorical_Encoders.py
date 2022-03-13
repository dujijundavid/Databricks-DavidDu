# Databricks notebook source
# MAGIC %md
# MAGIC # Categorical Encoders

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weight Of Evidence

# COMMAND ----------

"""Weight of Evidence"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class WOEEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Weight of Evidence coding for categorical features.

    Supported targets: binomial. For polynomial target support, see PolynomialWrapper.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    regularization: float
        the purpose of regularization is mostly to prevent division by zero.
        When regularization is 0, you may encounter division by zero.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = WOEEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Weight of Evidence (WOE) and Information Value Explained, from
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, regularization=1.0):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization
        self.feature_names = None

    # noinspection PyUnusedLocal
    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and binary y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Binary target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # Unite parameters into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        # The label must be binary with values {0,1}
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) + " unique value(s).")
        if y.isnull().any():
            raise ValueError("The target column y must not contain missing values.")
        if np.max(unique) < 1:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 1 was not found in the target.")
        if np.min(unique) > 0:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 0 was not found in the target.")

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))
        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data. When the data are used for model training,
        it is important to also pass the target in order to apply leave one out.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas types
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

            # Create a new column with regularized WOE.
            # Regularization helps to avoid division by zero.
            # Pre-calculate WOEs because logarithms are slow.
            nominator = (stats['sum'] + self.regularization) / (self._sum + 2*self.regularization)
            denominator = ((stats['count'] - stats['sum']) + self.regularization) / (self._count - self._sum + 2*self.regularization)
            woe = np.log(nominator / denominator)

            # Ignore unique values. This helps to prevent overfitting on id-like columns.
            woe[stats['count'] == 1] = 0

            if self.handle_unknown == 'return_nan':
                woe.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                woe.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                woe.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                woe.loc[-2] = 0

            # Store WOE for transform() function
            mapping[col] = woe

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Encoder

# COMMAND ----------

"""Target Encoder"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'chappers'


class TargetEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Target encoding for categorical features.
    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.
    For the case of categorical target: features are replaced with a blend of posterior probability of the target
    given particular categorical value and the prior probability of the target over all the training data.
    For the case of continuous target: features are replaced with a blend of the expected value of the target
    given particular categorical value and the expected value of the target over all the training data.
    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
    min_samples_leaf: int
        minimum samples to take category average into account.
    smoothing: float
        smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
        The value must be strictly bigger than 0.
    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = TargetEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None
    References
    ----------
    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, from
    https://dl.acm.org/citation.cfm?id=507538
    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value',
                     handle_unknown='value', min_samples_leaf=1, smoothing=1.0):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = float(smoothing)  # Make smoothing a float so that python 2 does not treat as integer division
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.feature_names = None

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index)

        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_target_encoding(X_ordinal, y)
        
        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))

        return self

    def fit_target_encoding(self, X, y):
        mapping = {}

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            prior = self._mean = y.mean()

            stats = y.groupby(X[col]).agg(['count', 'mean'])

            smoove = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.smoothing))
            smoothing = prior * (1 - smoove) + stats['mean'] * smoove
            smoothing[stats['count'] == 1] = prior

            if self.handle_unknown == 'return_nan':
                smoothing.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                smoothing.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                smoothing.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                smoothing.loc[-2] = prior

            mapping[col] = smoothing

        return mapping

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target info (such as transform test set)
            
        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.
        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not isinstance(self.feature_names, list):
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Ordinal Encoders

# COMMAND ----------

"""Ordinal or label encoding"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util
import warnings

__author__ = 'willmcginnis'


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as ordinal, in one ordered feature.

    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in; in this case, we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    mapping: list of dicts
        a mapping of class to label to use for the encoding, optional.
        the dict contains the keys 'col' and 'mapping'.
        the value of 'col' should be the feature name.
        the value of 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
        example mapping: [
            {'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}},
            {'col': 'col2', 'mapping': {None: 0, 'x': 1, 'y': 2}}
        ]
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
    handle_missing: str
        options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
        or -2 at transform time if nan is not a category during fit.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OrdinalEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(2)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf

    """

    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._dim = None
        self.feature_names = None

    @property
    def category_mapping(self):
        return self.mapping

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        _, categories = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )
        self.mapping = categories

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []

            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))

        return self

    def transform(self, X, override_return_df=False):
        """Perform the transformation to new categorical data.

        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinarily.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X if self.return_df else X.values

        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data. Will attempt best case reconstruction, which means
        it will return nan for handle_missing and handle_unknown settings that break the bijection. We issue
        warnings when some of those cases occur.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """

        # fail fast
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # first check the type and make deep copy
        X = util.convert_input(X_in, deep=True)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "be False when transforming the data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X if self.return_df else X.values

        if self.handle_unknown == 'value':
            for col in self.cols:
                if any(X[col] == -1):
                    warnings.warn("inverse_transform is not supported because transform impute "
                                  "the unknown category -1 when encode %s" % (col,))

        if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
            for col in self.cols:
                if X[col].isnull().any():
                    warnings.warn("inverse_transform is not supported because transform impute "
                                  "the unknown category nan when encode %s" % (col,))

        for switch in self.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, handle_unknown='value', handle_missing='value'):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """

        return_nan_series = pd.Series(data=[np.nan], index=[-2])

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                column = switch.get('col')
                col_mapping = switch['mapping']
                X[column] = X[column].map(col_mapping)

                if util.is_category(X[column].dtype):
                    if not isinstance(col_mapping, pd.Series):
                        col_mapping = pd.Series(col_mapping)
                    nan_identity = col_mapping.loc[col_mapping.index.isna()].values[0]
                    X[column] = X[column].cat.add_categories(nan_identity)
                    X[column] = X[column].fillna(nan_identity)

                try:
                    X[column] = X[column].astype(int)
                except ValueError as e:
                    X[column] = X[column].astype(float)

                if handle_unknown == 'value':
                    X[column].fillna(-1, inplace=True)
                elif handle_unknown == 'error':
                    missing = X[column].isnull()
                    if any(missing):
                        raise ValueError('Unexpected categories found in column %s' % column)

                if handle_missing == 'return_nan':
                    X[column] = X[column].map(return_nan_series).where(X[column] == -2, X[column])

        else:
            mapping_out = []
            for col in cols:

                nan_identity = np.nan

                categories = X[col].unique().tolist()
                if util.is_category(X[col].dtype):
                    # Avoid using pandas category dtype meta-data if possible, see #235, #238.
                    if X[col].dtype.ordered:
                        categories = [c for c in X[col].dtype.categories if c in categories]
                    if X[col].isna().any():
                        categories += [np.nan]

                index = pd.Series(categories).fillna(nan_identity).unique()

                data = pd.Series(index=index, data=range(1, len(index) + 1))

                if handle_missing == 'value' and ~data.index.isnull().any():
                    data.loc[nan_identity] = -2
                elif handle_missing == 'return_nan':
                    data.loc[nan_identity] = -2

                mapping_out.append({'col': col, 'mapping': data, 'data_type': X[col].dtype}, )

        return X, mapping_out

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names
          

# COMMAND ----------



class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# COMMAND ----------

# MAGIC %md
# MAGIC ## CatBoost Encoding

# COMMAND ----------

"""CatBoost coding"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class CatBoostEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """CatBoost coding for categorical features.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is very similar to leave-one-out encoding, but calculates the
    values "on-the-fly". Consequently, the values naturally vary
    during the training phase and it is not necessary to add random noise.

    Beware, the training data have to be randomly permutated. E.g.:

        # Random permutation
        perm = np.random.permutation(len(X))
        X = X.iloc[perm].reset_index(drop=True)
        y = y.iloc[perm].reset_index(drop=True)

    This is necessary because some data sets are sorted based on the target
    value and this coder encodes the features on-the-fly in a single pass.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
    sigma: float
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
        sigma gives the standard deviation (spread or "width") of the normal distribution.
    a: float
        additive smoothing (it is the same variable as "m" in m-probability estimate). By default set to 1.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = CatBoostEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Transforming categorical features to numerical features, from
    https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/

    .. [2] CatBoost: unbiased boosting with categorical features, from
    https://arxiv.org/abs/1706.09516

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, sigma=None, a=1):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.use_default_cols = cols is None  # if True, even a repeated call of fit() will select string columns from X
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.random_state = random_state
        self.sigma = sigma
        self.feature_names = None
        self.a = a

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.use_default_cols:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        categories = self._fit(
            X, y,
            cols=self.cols
        )
        self.mapping = categories

        X_temp = self.transform(X, y, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                          "Not found in generated cols.\n{}".format(e))

        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)



        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X
        X = self._transform(
            X, y,
            mapping=self.mapping
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _fit(self, X_in, y, cols=None):
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        self._mean = y.mean()

        return {col: self._fit_column_map(X[col], y) for col in cols}

    def _fit_column_map(self, series, y):
        category = pd.Categorical(series)

        categories = category.categories
        codes = category.codes.copy()

        codes[codes == -1] = len(categories)
        categories = np.append(categories, np.nan)

        return_map = pd.Series(dict([(code, category) for code, category in enumerate(categories)]))

        result = y.groupby(codes).agg(['sum', 'count'])
        return result.rename(return_map)

    def _transform(self, X_in, y, mapping=None):
        """
        The model uses a single column of floats to represent the means of the target variables.
        """
        X = X_in.copy(deep=True)
        random_state_ = check_random_state(self.random_state)

        # Prepare the data
        if y is not None:
            # Convert bools to numbers (the target must be summable)
            y = y.astype('double')

        for col, colmap in mapping.items():
            level_notunique = colmap['count'] > 1

            unique_train = colmap.index
            unseen_values = pd.Series([x for x in X_in[col].unique() if x not in unique_train], dtype=unique_train.dtype)

            is_nan = X_in[col].isnull()
            is_unknown_value = X_in[col].isin(unseen_values.dropna().astype(object))

            if self.handle_unknown == 'error' and is_unknown_value.any():
                raise ValueError('Columns to be encoded can not contain new values')

            if y is None:    # Replace level with its mean target; if level occurs only once, use global mean
                level_means = ((colmap['sum'] + self._mean) / (colmap['count'] + self.a)).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:
                # Simulation of CatBoost implementation, which calculates leave-one-out on the fly.
                # The nice thing about this is that it helps to prevent overfitting. The bad thing
                # is that CatBoost uses many iterations over the data. But we run just one iteration.
                # Still, it works better than leave-one-out without any noise.
                # See:
                #   https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
                # Cumsum does not work nicely with None (while cumcount does).
                # As a workaround, we cast the grouping column as string.
                # See: issue #209
                temp = y.groupby(X[col].astype(str)).agg(['cumsum', 'cumcount'])
                X[col] = (temp['cumsum'] - y + self._mean) / (temp['cumcount'] + self.a)

            if self.handle_unknown == 'value':
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype(float)
                X.loc[is_unknown_value, col] = self._mean
            elif self.handle_unknown == 'return_nan':
                X.loc[is_unknown_value, col] = np.nan

            if self.handle_missing == 'value':
                X.loc[is_nan & unseen_values.isnull().any(), col] = self._mean
            elif self.handle_missing == 'return_nan':
                X.loc[is_nan, col] = np.nan

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1., self.sigma, X[col].shape[0])

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """

        if not isinstance(self.feature_names, list):
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generalized linear mixed model Encoding

# COMMAND ----------

"""Generalized linear mixed model"""
import warnings
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.random import check_random_state
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as bgmm

__author__ = 'Jan Motl'


class GLMMEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Generalized linear mixed model.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is a supervised encoder similar to TargetEncoder or MEstimateEncoder, but there are some advantages:
    1) Solid statistical theory behind the technique. Mixed effects models are a mature branch of statistics.
    2) No hyper-parameters to tune. The amount of shrinkage is automatically determined through the estimation process.
    In short, the less observations a category has and/or the more the outcome varies for a category
    then the higher the regularization towards "the prior" or "grand mean".
    3) The technique is applicable for both continuous and binomial targets. If the target is continuous,
    the encoder returns regularized difference of the observation's category from the global mean.
    If the target is binomial, the encoder returns regularized log odds per category.

    In comparison to JamesSteinEstimator, this encoder utilizes generalized linear mixed models from statsmodels library.

    Note: This is an alpha implementation. The API of the method may change in the future.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    binomial_target: bool
        if True, the target must be binomial with values {0, 1} and Binomial mixed model is used.
        If False, the target must be continuous and Linear mixed model is used.
        If None (the default), a heuristic is applied to estimate the target type.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = GLMMEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Data Analysis Using Regression and Multilevel/Hierarchical Models, page 253, from
    https://faculty.psau.edu.sa/filedownload/doc-12-pdf-a1997d0d31f84d13c1cdc44ac39a8f2c-original.pdf

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, binomial_target=None):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.binomial_target = binomial_target
        self.feature_names = None

    # noinspection PyUnusedLocal
    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and binary y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Binary target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # Unite parameters into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))
        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        When the data are used for model training, it is important to also pass the target in order to apply leave one out.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)


        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas types
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Estimate target type, if necessary
        if self.binomial_target is None:
            if len(y.unique()) <= 2:
                binomial_target = True
            else:
                binomial_target = False
        else:
            binomial_target = self.binomial_target

        # The estimation does not have to converge -> at least converge to the same value.
        np.random.seed(2001)

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            data = self._rename_and_merge(X, y, col)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if binomial_target:
                        # Classification, returns (regularized) log odds per category as stored in vc_mean
                        # Note: md.predict() returns: output = fe_mean + vcp_mean + vc_mean[category]
                        md = bgmm.from_formula('target ~ 1', {'a': '0 + C(feature)'}, data).fit_vb()
                        index_names = [int(float(re.sub(r'C\(feature\)\[(\S+)\]', r'\1', index_name))) for index_name in md.model.vc_names]
                        estimate = pd.Series(md.vc_mean, index=index_names)
                    else:
                        # Regression, returns (regularized) mean deviation of the observation's category from the global mean
                        md = smf.mixedlm('target ~ 1', data, groups=data['feature']).fit()
                        tmp = dict()
                        for key, value in md.random_effects.items():
                            tmp[key] = value[0]
                        estimate = pd.Series(tmp)
            except np.linalg.LinAlgError:
                # Singular matrix -> just return all zeros
                estimate = pd.Series(np.zeros(len(values)), index=values)

            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            if len(X[col].unique()) == len(y):
                estimate[:] = 0

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = 0

            mapping[col] = estimate

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

    def _rename_and_merge(self, X, y, col):
        """
        Statsmodels requires:
            1) unique column names
            2) non-numeric columns names
        Solution: internally rename the columns.
        """
        merged = pd.DataFrame()
        merged['feature'] = X[col]
        merged['target'] = y

        return merged

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Encodings

# COMMAND ----------

def perform_WOE_encoding(df,encoding_columns,col="IsFraud"):
  y= df.col
  X= pd.DataFrame(df, columns=encoding_columns)
  WOE_enc = WOEEncoder(cols=encoding_columns).fit(X, y)
  WOE_encoded_df = WOE_enc.transform(X)
  print(WOE_encoded_df.info())
  return WOE_encoded_df

def perform_Target_encoding(df,encoding_columns,col="IsFraud"):
  X,y = df[encoding_columns],df[col]
  Target_enc = TargetEncoder(cols=encoding_columns,drop_invariant=False).fit(X,y)

  Target_encoding_results = Target_enc.transform(X)
  print(Target_encoding_results.info())
  return Target_encoding_results



def perform_GLMM_encoding(df,encoding_columns,col="IsFraud"):
  X,y = df[encoding_columns],df[col]
  GLMM_enc = GLMMEncoder(cols=encoding_columns,drop_invariant=False).fit(X,y)

  GLMM_encoding_results = GLMM_enc.transform(X)
  print(GLMM_encoding_results.info())
  return GLMM_encoding_results

def perform_CatBoost_encoding(df, encoding_columns,col="IsFraud"):
  X,y = df[encoding_columns],df[col]
  CatBpost_enc = CatBoostEncoder(cols=encoding_columns,drop_invariant=False).fit(X,y)

  CatBpost_encoding_results = CatBpost_enc.transform(X)
  print(CatBpost_encoding_results.info())
  return CatBpost_encoding_results



def perform_Ordinal_encoding(df, encoding_columns,col="IsFraud"):
  X,y = df[encoding_columns],df[col]
  Ordinal_enc = OrdinalEncoder(cols=encoding_columns,drop_invariant=False).fit(X,y)

  Ordinal_encoding_results = Ordinal_enc.transform(X)
  print(Ordinal_encoding_results.info())
  return Ordinal_encoding_results


# COMMAND ----------

# MAGIC %md
# MAGIC # Get Weight of Evidence & IV table

# COMMAND ----------


def get_WOE_IV_table(df,feature,target):
  df_woe_iv = (pd.crosstab(df[feature],df[target],
                        normalize='columns')
               .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
               .assign(iv=lambda dfx: np.sum(dfx['woe']*
                                             (dfx[1]-dfx[0]))))

  return df_woe_iv