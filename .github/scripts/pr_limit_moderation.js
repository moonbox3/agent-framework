// Copyright (c) Microsoft. All rights reserved.

function getPullRequest(context) {
  const pullRequest = context.payload.pull_request;
  if (!pullRequest?.number || !pullRequest.user?.login) {
    throw new Error('This script must be run from a pull_request_target event.');
  }

  return {
    author: pullRequest.user.login,
    number: pullRequest.number,
  };
}

async function ensureLabel({ github, owner, repo, labelName }) {
  try {
    await github.rest.issues.getLabel({
      owner,
      repo,
      name: labelName,
    });
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }

    try {
      await github.rest.issues.createLabel({
        owner,
        repo,
        name: labelName,
        color: 'd93f0b',
        description: 'Community author has exceeded the open pull request limit.',
      });
    } catch (createError) {
      if (createError.status !== 422) {
        throw createError;
      }
    }
  }
}

function buildLimitMessage({ author, maxOpenPrs, openPrCount }) {
  return [
    `Thank you for your contribution, @${author}.`,
    '',
    `To keep the review queue manageable, we currently limit community contributors to ${maxOpenPrs} `
      + `open pull requests at a time. This PR would put you at ${openPrCount} open pull requests, `
      + 'so we are closing it automatically.',
    '',
    'Please focus on getting your existing PRs reviewed, merged, or closed before opening another one. '
      + 'If a maintainer asked you to open this PR, a maintainer can reopen it.',
  ].join('\n');
}

async function getOpenPrCount({ github, owner, repo, author, pullRequestNumber }) {
  const query = `repo:${owner}/${repo} is:pr is:open author:${author}`;
  const response = await github.rest.search.issuesAndPullRequests({
    q: query,
    per_page: 100,
  });

  const indexedPrNumbers = response.data.items.map((item) => item.number);
  const currentPrIsIndexed = indexedPrNumbers.includes(pullRequestNumber);
  return currentPrIsIndexed ? response.data.total_count : response.data.total_count + 1;
}

async function enforcePrLimit({ github, context, core, maxOpenPrs, labelName }) {
  const { owner, repo } = context.repo;
  const { author, number } = getPullRequest(context);
  const openPrCount = await getOpenPrCount({
    github,
    owner,
    repo,
    author,
    pullRequestNumber: number,
  });

  if (openPrCount <= maxOpenPrs) {
    core.info(
      `${author} has ${openPrCount} open pull request(s), which is within the limit of ${maxOpenPrs}.`,
    );
    return {
      author,
      closed: false,
      openPrCount,
    };
  }

  await ensureLabel({
    github,
    owner,
    repo,
    labelName,
  });

  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: number,
    labels: [labelName],
  });

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: number,
    body: buildLimitMessage({
      author,
      maxOpenPrs,
      openPrCount,
    }),
  });

  await github.rest.pulls.update({
    owner,
    repo,
    pull_number: number,
    state: 'closed',
  });

  core.info(
    `${author} has ${openPrCount} open pull request(s), which exceeds the limit of ${maxOpenPrs}. `
      + `Closed PR #${number}.`,
  );

  return {
    author,
    closed: true,
    openPrCount,
  };
}

module.exports = {
  buildLimitMessage,
  enforcePrLimit,
  getOpenPrCount,
};
