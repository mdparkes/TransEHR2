"""Number and string formatting that follows JMIR house style.

The rules implemented here come from JMIR Publications' "Guidelines for
Reporting Statistics":

* Numerical values less than 1 require a leading zero, except for P
  values, alpha levels and beta values.
* Use an en dash (or minus sign) to indicate negative values.
* Do not use spaces around signs of equality or inequality.
* Exact P values are reported to 2 decimal places, with three
  exceptions: P<.01 takes 3 decimal places, rounding that would change
  the significance level takes 3 decimal places, and P values below
  .001 or above .99 are reported as inequalities.
* P values can never equal 0 or 1.
* When several statistics appear inside one set of parentheses, related
  statistics are separated by commas and unrelated statistics by
  semicolons.
"""

import math

# U+2013 EN DASH, used for negative values in running text and tables.
EN_DASH = '–'


def fmt_number(value, precision=3, use_en_dash=True):
    """Format a numeric value with a leading zero and an en dash.

    Args:
        value: The value to format. ``None`` and NaN render as an em dash
            placeholder.
        precision: Number of decimal places.
        use_en_dash: Render negative values with an en dash rather than a
            hyphen-minus.

    Returns:
        The formatted value as a string.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'

    text = f'{abs(value):.{precision}f}'
    if value < 0 and float(text) != 0.0:
        return (EN_DASH if use_en_dash else '-') + text
    return text


def fmt_p_value(p, alpha=0.05):
    """Format a P value according to JMIR rules.

    P values are reported to 2 decimal places, except that 3 decimal
    places are used when P<.01 or when rounding to 2 decimal places
    would move the value across the significance level. Values below
    .001 and above .99 are reported as inequalities, because a P value
    can be neither 0 nor 1. The leading zero is omitted.

    Args:
        p: The P value, or ``None``/NaN when the test was not applicable.
        alpha: The study's significance level, used to detect the case
            where rounding would change the apparent significance.

    Returns:
        A string such as ``'P=.03'``, ``'P=.048'``, ``'P<.001'`` or
        ``'P>.99'``. Returns an em dash when ``p`` is unavailable.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '—'

    if p < 0.001:
        return 'P<.001'
    if p > 0.99:
        return 'P>.99'

    decimals = 2
    if p < 0.01:
        decimals = 3
    elif (p < alpha) != (round(p, 2) < alpha):
        # Rounding to 2 decimal places would change the significance
        # level, e.g. P=.048 must not be reported as P=.05.
        decimals = 3

    text = f'{p:.{decimals}f}'
    if decimals == 3 and text.endswith('0'):
        # .010 reads better as .01; the value is unchanged.
        text = text[:-1]
    return 'P=' + text.lstrip('0')


def fmt_mean_se(mean, se, precision=3):
    """Format a mean together with its standard error.

    JMIR requires the measure of variability to be named rather than
    implied by a plus-or-minus sign, following the pattern
    "Mean systolic blood pressure was 128 (SD 12) mm Hg".

    Args:
        mean: The sample mean across folds.
        se: The standard error of the mean.
        precision: Number of decimal places for both values.

    Returns:
        A string such as ``'0.836 (SE 0.005)'``.
    """
    return (f'{fmt_number(mean, precision)} '
            f'(SE {fmt_number(se, precision)})')


def fmt_cell(mean, se, p=None, precision=3, alpha=0.05):
    """Format a full table cell: a mean, its SE and an optional P value.

    Related statistics inside the parentheses are separated by a
    semicolon because the SE and the P value are unrelated quantities.

    Args:
        mean: The sample mean across folds.
        se: The standard error of the mean.
        p: The P value for the comparison against the control, or
            ``None`` for the control column itself.
        precision: Number of decimal places for the mean and SE.
        alpha: The study's significance level.

    Returns:
        A string such as ``'0.845 (SE 0.004; P=.03)'``.
    """
    body = f'SE {fmt_number(se, precision)}'
    if p is not None:
        body += '; ' + fmt_p_value(p, alpha)
    return f'{fmt_number(mean, precision)} ({body})'


def fmt_t_statistic(t, df, precision=2):
    """Format a t statistic with its degrees of freedom, JMIR table style.

    In tables JMIR presents degrees of freedom in parentheses after the
    test statistic, e.g. a "t test (df)" column containing "2.68 (15)".

    Args:
        t: The test statistic.
        df: Degrees of freedom.
        precision: Number of decimal places for the statistic.

    Returns:
        A string such as ``'2.68 (4)'``.
    """
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return '—'
    if isinstance(t, float) and math.isinf(t):
        return ('–' if t < 0 else '') + f'∞ ({df})'
    return f'{fmt_number(t, precision)} ({df})'


def sentence_case(text):
    """Lower-case all but the first character, preserving embedded caps.

    JMIR requires column and row headings in sentence case. Acronyms and
    model names that are already capitalised are left alone, so only the
    very first character is adjusted and only when the word that
    contains it is not itself an acronym.

    Args:
        text: The heading text.

    Returns:
        The heading in sentence case.
    """
    if not text:
        return text
    first_word = text.split(' ', 1)[0]
    if first_word.isupper() and len(first_word) > 1:
        return text
    return text[0].upper() + text[1:]
