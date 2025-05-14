let currentSectionIndex = 0;
let totalScore = 0;

const sectionOrder = ["Age", "Visa/Residency Status", "Duration in Ireland"];
const chatbox = document.getElementById("chatbox");
const inputField = document.getElementById("userInput");
const stepIndicator = document.getElementById("current-step");
const totalSteps = document.getElementById("total-steps");
const typingIndicator = document.getElementById("typing-indicator");

totalSteps.textContent = sectionOrder.length;

window.onload = () => {
  displayMessage("👋 Welcome to the Loan Evaluation Assistant!", "bot");
  triggerBackendForQuestion();
};

// Event listener for 'Enter' key press to send message
inputField.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault(); // Prevent the default Enter behavior (new line)
    sendMessage();
  }
});

async function sendMessage() {
  const userText = inputField.value.trim();
  if (!userText) return;

  displayMessage(userText, "user");
  inputField.value = "";

  const currentSection = sectionOrder[currentSectionIndex];

  try {
    showTypingIndicator(true);

    const response = await fetch("https://solid-guacamole-vwwr5wv76v7fx7gg-5000.app.github.dev/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        section: currentSection,
        question: currentSection,
        answer: userText,
      }),
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    totalScore += data.score;

    const botMessage = `✅ **${data.section}**\nAnswer: ${data.answer}\nScore: ${data.score}\nReason: ${data.reason}\nEligibility: ${data.eligability}`;
    displayMessage(botMessage, "bot");

    currentSectionIndex++;
    stepIndicator.textContent = currentSectionIndex + 1;

    if (currentSectionIndex < sectionOrder.length) {
      triggerBackendForQuestion();
    } else {
      displayMessage(
        totalScore > 50 ? "🎉 Loan Approved!" : "❌ Loan Denied.",
        "bot"
      );
      displayMessage(`Final Score: ${totalScore}`, "bot");
    }
  } catch (error) {
    displayMessage("⚠️ Error connecting to the server. Please try again later.", "bot");
    console.error(error);
  } finally {
    showTypingIndicator(false);
  }
}

async function triggerBackendForQuestion() {
  const currentSection = sectionOrder[currentSectionIndex];

  try {
    showTypingIndicator(true);

    const response = await fetch("https://solid-guacamole-vwwr5wv76v7fx7gg-5000.app.github.dev/get_question", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section: currentSection }),
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    displayMessage(data.question, "bot");
  } catch (error) {
    displayMessage("⚠️ Failed to load the question.", "bot");
    console.error(error);
  } finally {
    showTypingIndicator(false);
  }
}

function displayMessage(message, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}`;
  messageDiv.innerText = message;
  chatbox.appendChild(messageDiv);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function showTypingIndicator(show) {
  typingIndicator.style.display = show ? "block" : "none";
}
