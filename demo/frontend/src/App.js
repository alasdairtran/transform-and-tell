import React, { Component } from 'react';
import axios from 'axios';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import ReactPlayer from 'react-player';
import * as d3 from 'd3';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      articleURL: '',
      imagePosition: 0,
      isLoaded: false,
      isLoading: false,
      isScraped: false,
      isScraping: false,
      article: null,
      title: '',
      sections: [],
      imageURL: '',
      imageURLs: [],
      start: '',
      before: '',
      after: '',
      trueCaption: '',
      generatedCaption: '',
      attns: [],
      image: null,
      hasError: false,
      errorMessage: '',
      showModal: false,
    };
    this.buttonRef = React.createRef();
  }

  componentDidMount() {
    document.body.classList.add('bg-light');
  }

  componentDidUpdate() {
    if (this.state.isScraped && ~this.state.isLoaded) {
      this.buttonRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
        inline: 'center',
      });
    }
  }

  scrapeArticle = (e) => {
    e.preventDefault();
    this.setState({
      isScraped: false,
      isScraping: true,
      isLoaded: false,
      hasError: false,
      imagePosition: 0,
    });
    const query = {
      url: this.state.articleURL,
    };
    axios
      .post('/api/scrape/', query)
      .then((res) => {
        if (res.data.error) {
          this.setState({
            isScraping: false,
            hasError: true,
            isScraped: false,
            errorMessage: res.data.error,
          });
        } else {
          this.setState({
            isScraped: true,
            isScraping: false,
            imageURLs: res.data.image_urls,
            sections: res.data.sections,
            title: res.data.title,
          });
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };

  fetchCaption = (e) => {
    e.preventDefault();
    this.setState({ isLoaded: false, isLoading: true, hasError: false });

    // Remove unwated images. Can't send to server requests that are too big.
    const sections = [];
    let cursor = 0;
    this.state.sections.forEach((s) => {
      if (s.type === 'paragraph') {
        sections.push(s);
      } else if (cursor === this.state.imagePosition) {
        sections.push(s);
        cursor += 1;
      } else {
        cursor += 1;
      }
    });

    const query = {
      sections: sections,
      title: this.state.title,
      pos: 0,
    };
    axios
      .post('/api/caption/', query)
      .then((res) => {
        if (res.data.error) {
          this.setState({
            isLoading: false,
            hasError: true,
            isLoaded: false,
            errorMessage: res.data.error,
          });
        } else {
          this.setState({
            isLoaded: true,
            isLoading: false,
            title: res.data.title,
            imageURL: res.data.image_url,
            start: res.data.start,
            before: res.data.before,
            after: res.data.after,
            trueCaption: res.data.true_caption,
            generatedCaption: res.data.generated_caption,
            attns: res.data.attns,
            image: res.data.image,
          });
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };

  handleURLChange = (e) => {
    this.setState({
      articleURL: e.target.value,
      isScraped: false,
      isLoaded: false,
    });
  };

  selectArticle = (e) => {
    this.setState({
      articleURL: e.target.getAttribute('url'),
      isScraped: false,
      isLoaded: false,
    });
  };

  showModal = (e) => {
    this.setState({ showModal: true });
  };

  handleClose = (e) => {
    this.setState({ showModal: false });
  };

  handleImagePositionChange = (index, e) => {
    this.setState({ imagePosition: index, isLoaded: false });
  };

  render() {
    const examples = [
      {
        title: 'Two Crises Convulse a Nation: A Pandemic and Police Violence',
        url:
          'https://www.nytimes.com/2020/05/31/us/george-floyd-protests-coronavirus.html',
      },
      {
        title:
          'Testing Is Key to Beating Coronavirus, Right? Japan Has Other Ideas',
        url:
          'https://www.nytimes.com/2020/05/29/world/asia/japan-coronavirus.html',
      },
      {
        title:
          "'Turn Off the Sunshine': Why Shade Is a Mark of Privilege in Los Angeles",
        url:
          'https://www.nytimes.com/2019/12/01/us/los-angeles-shade-climate-change.html',
      },
      {
        title: 'Ready, Set, Ski! In China, Snow Sports are the Next Big Thing',
        url:
          'https://www.nytimes.com/2019/11/27/travel/Skiing-in-China-Olympics.html',
      },
      {
        title:
          'New Strawberry-Flavored H.I.V. Drugs for Babies Are Offered at $1 a Day',
        url:
          'https://www.nytimes.com/2019/11/29/health/AIDS-drugs-children.html',
      },
    ];

    return (
      <div className="container">
        <Modal show={this.state.showModal} onHide={this.handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Abstract</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            We propose an end-to-end model which generates captions for images
            embedded in news articles. News images present two key challenges:
            they rely on real-world knowledge, especially about named entities;
            and they typically have linguistically rich captions that include
            uncommon words. We address the first challenge by associating words
            in the caption with faces and objects in the image, via a
            multi-modal, multi-head attention mechanism. We tackle the second
            challenge with a state-of-the-art transformer language model that
            uses byte-pair-encoding to generate captions as a sequence of word
            parts. On the GoodNews dataset, our model outperforms the previous
            state of the art by a factor of four in CIDEr score (13 to 54). This
            performance gain comes from a unique combination of language models,
            word representation, image embeddings, face embeddings, object
            embeddings, and improvements in neural network design. We also
            introduce the NYTimes800k dataset which is 70% larger than GoodNews,
            has higher article quality, and includes the locations of images
            within articles as an additional contextual cue.
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={this.handleClose}>
              Close
            </Button>
          </Modal.Footer>
        </Modal>

        <div className="py-5">
          <h2 className="text-center">
            Transform and Tell: Entity-Aware News Image Captioning
          </h2>
          <p className="lead text-center">
            <p>
              <i>Alasdair Tran, Alexander Mathews, Lexing Xie</i>
            </p>
            <a href="#abstractModal" onClick={this.showModal}>
              Abstract
            </a>
            &nbsp;|&nbsp;
            <a href="https://arxiv.org/abs/2004.08070">Paper</a>&nbsp;|&nbsp;
            <a href="https://github.com/alasdairtran/transform-and-tell">
              GitHub
            </a>
            &nbsp;|&nbsp;
            <a href="http://cm.cecs.anu.edu.au/post/transform_and_tell/">
              Blog
            </a>
          </p>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ReactPlayer url="https://www.youtube.com/watch?v=lei1VOJbf40yarn " />
          </div>
          <br />

          <p>
            Transform and Tell is a captioning model that takes a news image and
            generate a caption for it using information from the article, with a
            special focus on faces and names. To see it in action, click on one
            of the following examples:
          </p>

          <div className="list-group">
            {examples.map((example, index) => (
              <button
                key={index}
                type="button"
                className={
                  'list-group-item list-group-item-action' +
                  (this.state.articleURL === example.url
                    ? ' list-group-item-secondary'
                    : '')
                }
                onClick={this.selectArticle}
                url={example.url}
              >
                {example.title}
              </button>
            ))}
          </div>
          <br />
          <form>
            <div className="form-group">
              <label htmlFor="articleURL">
                Or manually provide the URL to a New York Times article:
              </label>
              <input
                type="url"
                className="form-control"
                id="articleURL"
                aria-describedby="urlHelp"
                placeholder="Article URL"
                value={this.state.articleURL}
                onChange={this.handleURLChange}
              />
            </div>
            <button
              type="submit"
              className="btn btn-lg btn-primary"
              onClick={this.scrapeArticle}
              disabled={this.state.isScraping}
            >
              {this.state.isScraping ? 'Scraping Article...' : 'Scrape Article'}
            </button>

            {this.state.isScraped && (
              <div className="my-2">
                <p>Choose an image to caption:</p>
                <div className="row">
                  {this.state.imageURLs.map((url, index) => (
                    <div key={index} className="col-md-2 mb-2">
                      <img
                        className={'img-thumbnail'}
                        style={
                          this.state.imagePosition === index
                            ? {
                                border: '5px solid black',
                              }
                            : {}
                        }
                        src={url}
                        key={index}
                        alt=""
                        onClick={this.handleImagePositionChange.bind(
                          this,
                          index
                        )}
                      />
                    </div>
                  ))}
                </div>
                <button
                  ref={this.buttonRef}
                  type="submit"
                  className="btn btn-lg btn-primary"
                  onClick={this.fetchCaption}
                  disabled={this.state.isLoading}
                >
                  {this.state.isLoading
                    ? 'Running Model...'
                    : 'Generate Caption'}
                </button>
              </div>
            )}
          </form>
        </div>

        {this.state.hasError && (
          <div className="alert alert-danger" role="alert">
            {this.state.errorMessage}
          </div>
        )}
        {this.state.isLoaded && (
          <Generation
            title={this.state.title}
            start={this.state.start}
            before={this.state.before}
            after={this.state.after}
            imageURL={this.state.imageURL}
            generatedCaption={this.state.generatedCaption}
            trueCaption={this.state.trueCaption}
            attns={this.state.attns}
            image={this.state.image}
          />
        )}
      </div>
    );
  }
}

class Generation extends Component {
  constructor(props) {
    super(props);
    this.captionRef = React.createRef();
    this.svg = null;
    this.margin = { top: 10, right: 10, bottom: 10, left: 10 };
    this.width = 224;
    this.height = 224;
    this.state = {
      selectedWordIdx: null,
    };
  }
  componentDidMount(newProps) {
    this.captionRef.current.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
      inline: 'center',
    });

    this.svg = d3
      .select(this.refs.imageCanvas)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr(
        'transform',
        'translate(' + this.margin.left + ',' + this.margin.top + ')'
      );
    this.svg
      .append('svg:image')
      .attr('xlink:href', `data:image/jpeg;base64,${this.props.image}`);
  }

  formatTextBlock = (words) => {
    let myColor = d3
      .scaleLinear()
      .range(['#f7f7f7', '#8340A0'])
      .domain([0, 0.2]);

    return words.map((item, key) => {
      if (item.text === '\n') {
        return key === 0 ? (
          <span key={key} />
        ) : (
          <div key={key}>
            <br />
          </div>
        );
      }
      return (
        <span
          key={key}
          style={{
            color: 'black',
            backgroundColor:
              this.state.selectedWordIdx !== null
                ? myColor(item.attns.reduce((a, b) => a + b, 0))
                : '#f7f7f7',
          }}
        >
          {item.text}
        </span>
      );
    });
  };

  highlightArticle = () => {
    const k =
      this.state.selectedWordIdx !== null ? this.state.selectedWordIdx : 0;
    const a = this.props.attns[k].attns.article;

    let cursor = 0;
    const title_words = [];
    const article_words = [];
    a.forEach((item, key) => {
      if (item.text === '\n') {
        cursor++;
      }

      if (cursor === 0) {
        title_words.push(item);
      } else {
        article_words.push(item);
      }
    });

    return (
      <div>
        <h4>{this.formatTextBlock(title_words)}</h4>

        <div className="mb-3">
          <div style={{ float: 'left' }} ref="imageCanvas" />{' '}
          {this.formatTextBlock(article_words)}
        </div>
      </div>
    );
  };

  selectWord = (a, idx) => {
    this.setState({
      selectedWordIdx: idx,
    });

    const img_attn = a.attns.image;
    const avg_img_attn = [];
    // Let's average across all layers for now
    // We'll ignore the last two since that's attention on nothing.
    for (var i = 0; i < img_attn[0].length - 2; i++) {
      let s = 0.0;
      for (var j = 0; j < img_attn.length; j++) {
        s += img_attn[j][i];
      }
      avg_img_attn.push({
        x: i % 7,
        y: Math.floor(i / 7),
        value: (100 * s) / img_attn.length,
      });
    }

    // Remove existing svg
    d3.selectAll('rect').remove();

    // set the dimensions and margins of the graph

    // append the svg object to the body of the page

    // Build color scale
    var myOpacity = d3.scaleLinear().range([0.6, 0]).domain([1, 4]);

    const domain = [0, 1, 2, 3, 4, 5, 6];
    const x = d3.scaleBand().range([0, this.width]).domain(domain).padding(0);
    const y = d3.scaleBand().range([0, this.height]).domain(domain).padding(0);

    //Read the data
    this.svg
      .selectAll()
      .data(avg_img_attn)
      .enter()
      .append('rect')
      .attr('x', function (d) {
        return x(d.x);
      })
      .attr('y', function (d) {
        return y(d.y);
      })
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .style('fill', 'black')
      .style('opacity', function (d) {
        return idx === null ? 0 : myOpacity(d.value);
      });
  };

  render() {
    return (
      <div className="row" ref={this.captionRef}>
        <div className="alert alert-success">
          <h4 className="mb-3">Generated caption</h4>
          <div className="mb-3">
            {this.props.attns.map((a, idx) => {
              return (
                <button
                  key={idx}
                  type="button"
                  className={`btn ${
                    this.state.selectedWordIdx === idx
                      ? 'btn-dark'
                      : 'btn-outline-dark'
                  }`}
                  onClick={this.selectWord.bind(this, a, idx)}
                  onMouseOver={this.selectWord.bind(this, a, idx)}
                  onMouseOut={this.selectWord.bind(this, a, null)}
                >
                  {a.tokens}
                </button>
              );
            })}
          </div>
          <p>
            <i>
              Hover over a word in the caption to see the attention scores over
              the contexts below. More attention is paid to words highlighted
              with a darker purlple and to image regions more lightly shaded.
            </i>
          </p>
          <hr />
          <p>
            <strong>Ground-truth caption: </strong>
            {this.props.trueCaption}
          </p>
        </div>
        <div className="mb-3">{this.highlightArticle()}</div>
      </div>
    );
  }
}

export default App;
