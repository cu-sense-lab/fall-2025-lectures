import io
import numpy as np
import numba as nb


def process_sample_stream(f: io.BufferedReader):

    # create sample buffer
    # for each item in queue, 


# Alternatively, in main tracking loop
# We can load samples, call schedule next correlation on each channel
# Then run scheduled correlations?

# for channel in channels:
#  correlator.schedule_correlation(channel, samples)
# with file:
#  while True:
#   load samples
#   for corr in queue:
#    if ready:
#     run correlation
#     if complete:
#      channel.process result
#    elif no longer valid:
#     channel.process result (None)


# Sample stream can expose readonly sample blocks
# But it makes sense to define sample preprocessing before they go into the buffer
# So what we need is a way to specify a sample source, and a set of processes to apply to the samples
# before they are stored in the sample stream buffer
# The sample stream buffer can then provide sample blocks based on the request queue

# Sample Source
# Sample Preprocessor
# Sample Stream

# We don't want to write a high performance configurable preprocessor
# So we will write a straightforward rigid one
# In GNSS signal processing, our samples need to potentially be mixed down,
# downsampled, and filtered.

# How do we ensure that the sample stream does not override samples that are still in use?
# We do this by ensuring that the circular buffer is 3X (or 4X?) the max requested block length
# This doesnt solve the problem if one channel is taking a particularly long time to process a block.
# This is maybe why we need channels to request correlation instead of the sample block directly.

